#include "abp_core.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#pragma pack(push, 1)
typedef struct TraceRecord {
    uint32_t t;
    uint8_t action;
    uint16_t dt;
    float reward;
    uint8_t terminated;
    uint8_t truncated;
    uint8_t invalid_action;
    int16_t resolved_action;
    float obs[ABP_OBS_DIM];
    float info_selected[13];
} TraceRecord;
#pragma pack(pop)

static void print_usage(const char *program_name) {
    fprintf(stderr, "Usage: %s --seed <seed> --actions <actions.bin> --out <trace.bin>\n",
            program_name);
}

static int read_actions(const char *path, uint8_t **actions_out, size_t *count_out) {
    FILE *fp = NULL;
    long size = 0;
    uint8_t *buffer = NULL;

    fp = fopen(path, "rb");
    if (fp == NULL) {
        fprintf(stderr, "Failed to open actions file '%s': %s\n", path, strerror(errno));
        return 0;
    }

    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return 0;
    }

    size = ftell(fp);
    if (size < 0) {
        fclose(fp);
        return 0;
    }

    if (fseek(fp, 0, SEEK_SET) != 0) {
        fclose(fp);
        return 0;
    }

    if (size == 0) {
        fclose(fp);
        *actions_out = NULL;
        *count_out = 0;
        return 1;
    }

    buffer = (uint8_t *)malloc((size_t)size);
    if (buffer == NULL) {
        fclose(fp);
        return 0;
    }

    if (fread(buffer, 1, (size_t)size, fp) != (size_t)size) {
        free(buffer);
        fclose(fp);
        return 0;
    }

    fclose(fp);
    *actions_out = buffer;
    *count_out = (size_t)size;
    return 1;
}

int main(int argc, char **argv) {
    const char *actions_path = NULL;
    const char *output_path = NULL;
    uint64_t seed = 0;

    uint8_t *actions = NULL;
    size_t action_count = 0;
    FILE *out_file = NULL;

    AbpCoreState *state = NULL;
    AbpCoreStepResult step_result;

    size_t i = 0;
    uint64_t episode_index = 0;

    for (i = 1; i + 1 < (size_t)argc; i += 2) {
        if (strcmp(argv[i], "--seed") == 0) {
            seed = (uint64_t)_strtoui64(argv[i + 1], NULL, 10);
        } else if (strcmp(argv[i], "--actions") == 0) {
            actions_path = argv[i + 1];
        } else if (strcmp(argv[i], "--out") == 0) {
            output_path = argv[i + 1];
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    if (actions_path == NULL || output_path == NULL) {
        print_usage(argv[0]);
        return 1;
    }

    if (!read_actions(actions_path, &actions, &action_count)) {
        fprintf(stderr, "Failed to read actions from '%s'\n", actions_path);
        return 1;
    }

    out_file = fopen(output_path, "wb");
    if (out_file == NULL) {
        fprintf(stderr, "Failed to open output file '%s': %s\n", output_path, strerror(errno));
        free(actions);
        return 1;
    }

    state = abp_core_create(NULL, seed);
    if (state == NULL) {
        fprintf(stderr, "Failed to initialize core state\n");
        fclose(out_file);
        free(actions);
        return 1;
    }

    for (i = 0; i < action_count; ++i) {
        TraceRecord record;

        abp_core_step(state, actions[i], &step_result);

        memset(&record, 0, sizeof(record));
        record.t = (uint32_t)i;
        record.action = actions[i];
        record.dt = step_result.dt;
        record.reward = step_result.reward;
        record.terminated = step_result.terminated;
        record.truncated = step_result.truncated;
        record.invalid_action = step_result.invalid_action;
        record.resolved_action = step_result.action;
        memcpy(record.obs, step_result.obs, sizeof(step_result.obs));

        record.info_selected[0] = step_result.credits;
        record.info_selected[1] = step_result.net_profit;
        record.info_selected[2] = step_result.profit_per_tick;
        record.info_selected[3] = step_result.survival;
        record.info_selected[4] = step_result.overheat_ticks;
        record.info_selected[5] = step_result.pirate_encounters;
        record.info_selected[6] = step_result.value_lost_to_pirates;
        record.info_selected[7] = step_result.fuel_used;
        record.info_selected[8] = step_result.hull_damage;
        record.info_selected[9] = step_result.tool_wear;
        record.info_selected[10] = step_result.scan_count;
        record.info_selected[11] = step_result.mining_ticks;
        record.info_selected[12] = step_result.cargo_utilization_avg;

        if (fwrite(&record, sizeof(record), 1, out_file) != 1) {
            fprintf(stderr, "Failed writing trace output\n");
            abp_core_destroy(state);
            fclose(out_file);
            free(actions);
            return 1;
        }

        if (step_result.terminated || step_result.truncated) {
            ++episode_index;
            abp_core_reset(state, seed + episode_index, NULL);
        }
    }

    abp_core_destroy(state);
    fclose(out_file);
    free(actions);

    return 0;
}
