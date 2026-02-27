#ifndef ABP_CORE_H
#define ABP_CORE_H

#include <stdint.h>

#include "abp_rng.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ABP_N_COMMODITIES 6
#define ABP_MAX_NODES 32
#define ABP_MAX_NEIGHBORS 6
#define ABP_MAX_ASTEROIDS 16
#define ABP_NODE_TYPES 3
#define ABP_NODE_STATION 0
#define ABP_NODE_CLUSTER 1
#define ABP_NODE_HAZARD 2

#define ABP_OBS_DIM 260
#define ABP_N_ACTIONS 69

#define ABP_FUEL_MAX 1000.0f
#define ABP_HULL_MAX 100.0f
#define ABP_HEAT_MAX 100.0f
#define ABP_TOOL_MAX 100.0f
#define ABP_CARGO_MAX 200.0f
#define ABP_ALERT_MAX 100.0f
#define ABP_TIME_MAX 20000.0f

typedef struct AbpCoreConfig {
    float time_max;
    float invalid_action_penalty;
} AbpCoreConfig;

typedef struct AbpCoreStepResult {
    float obs[ABP_OBS_DIM];
    float reward;
    uint8_t terminated;
    uint8_t truncated;
    uint8_t invalid_action;
    uint16_t dt;
} AbpCoreStepResult;

typedef struct AbpCoreState {
    AbpCoreConfig config;
    AbpRng rng;
    uint64_t seed;

    uint32_t ticks_elapsed;
    float time_remaining;

    uint8_t current_node;
    int8_t selected_asteroid;

    float credits;
    float fuel;
    float hull;
    float heat;
    float tool_condition;
    float alert;
    float cargo[ABP_N_COMMODITIES];

    uint8_t node_type[ABP_MAX_NODES];
    int8_t neighbors[ABP_MAX_NODES][ABP_MAX_NEIGHBORS];

    float market_price[ABP_N_COMMODITIES];
    float market_prev_price[ABP_N_COMMODITIES];

    float obs_buffer[ABP_OBS_DIM];
} AbpCoreState;

void abp_core_default_config(AbpCoreConfig *config);

AbpCoreState *abp_core_create(const AbpCoreConfig *config, uint64_t seed);
void abp_core_destroy(AbpCoreState *state);

void abp_core_init(AbpCoreState *state, const AbpCoreConfig *config, uint64_t seed);
void abp_core_reset(AbpCoreState *state, uint64_t seed, float *obs_out);
void abp_core_step(AbpCoreState *state, uint8_t action, AbpCoreStepResult *out);

#ifdef __cplusplus
}
#endif

#endif
