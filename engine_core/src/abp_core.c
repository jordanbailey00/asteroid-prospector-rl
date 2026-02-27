#include "abp_core.h"

#include <string.h>

static void abp_pack_obs(AbpCoreState *state, float *obs_out) {
    float fuel_sum = 0.0f;
    float credits_norm = 0.0f;
    uint32_t c_idx = 0;

    memset(state->obs_buffer, 0, sizeof(state->obs_buffer));

    for (c_idx = 0; c_idx < ABP_N_COMMODITIES; ++c_idx) {
        fuel_sum += state->cargo[c_idx];
    }

    if (state->credits > 0.0f) {
        credits_norm = state->credits / 1000000.0f;
        if (credits_norm > 1.0f) {
            credits_norm = 1.0f;
        }
    }

    state->obs_buffer[0] = state->fuel / ABP_FUEL_MAX;
    state->obs_buffer[1] = state->hull / ABP_HULL_MAX;
    state->obs_buffer[2] = state->heat / ABP_HEAT_MAX;
    state->obs_buffer[3] = state->tool_condition / ABP_TOOL_MAX;
    state->obs_buffer[4] = fuel_sum / ABP_CARGO_MAX;
    state->obs_buffer[5] = state->alert / ABP_ALERT_MAX;
    state->obs_buffer[6] = state->time_remaining / state->config.time_max;
    state->obs_buffer[7] = credits_norm;

    for (c_idx = 0; c_idx < ABP_N_COMMODITIES; ++c_idx) {
        state->obs_buffer[8u + c_idx] = state->cargo[c_idx] / ABP_CARGO_MAX;
    }

    if (state->current_node == 0u) {
        state->obs_buffer[17] = 1.0f;
        state->obs_buffer[19] = 1.0f;
    } else {
        state->obs_buffer[20] = 1.0f;
    }

    for (c_idx = 0; c_idx < ABP_N_COMMODITIES; ++c_idx) {
        state->obs_buffer[244u + c_idx] = state->market_price[c_idx] / 300.0f;
        state->obs_buffer[250u + c_idx] =
            (state->market_price[c_idx] - state->market_prev_price[c_idx]) / 100.0f;
    }

    if (obs_out != NULL) {
        memcpy(obs_out, state->obs_buffer, sizeof(state->obs_buffer));
    }
}

static void abp_seed_world_stub(AbpCoreState *state) {
    uint32_t node = 0;
    uint32_t slot = 0;
    uint32_t commodity = 0;

    for (node = 0; node < ABP_MAX_NODES; ++node) {
        state->node_type[node] = ABP_NODE_CLUSTER;
        for (slot = 0; slot < ABP_MAX_NEIGHBORS; ++slot) {
            state->neighbors[node][slot] = -1;
        }
    }

    state->node_type[0] = ABP_NODE_STATION;

    for (node = 1; node < 8; ++node) {
        state->node_type[node] =
            (abp_rng_next_u32(&state->rng) % 5u == 0u) ? ABP_NODE_HAZARD : ABP_NODE_CLUSTER;

        state->neighbors[node - 1u][0] = (int8_t)node;
        state->neighbors[node][1] = (int8_t)(node - 1u);
    }

    for (commodity = 0; commodity < ABP_N_COMMODITIES; ++commodity) {
        float base = 60.0f + 25.0f * (float)commodity;
        float jitter = 40.0f * abp_rng_next_f32(&state->rng);
        state->market_price[commodity] = base + jitter;
        state->market_prev_price[commodity] = state->market_price[commodity];
    }
}

void abp_core_default_config(AbpCoreConfig *config) {
    if (config == NULL) {
        return;
    }

    config->time_max = ABP_TIME_MAX;
    config->invalid_action_penalty = 0.01f;
}

void abp_core_init(AbpCoreState *state, const AbpCoreConfig *config, uint64_t seed) {
    uint32_t c_idx = 0;

    memset(state, 0, sizeof(*state));

    if (config != NULL) {
        state->config = *config;
    } else {
        abp_core_default_config(&state->config);
    }

    state->seed = seed;
    state->current_node = 0u;
    state->selected_asteroid = -1;

    state->fuel = ABP_FUEL_MAX;
    state->hull = ABP_HULL_MAX;
    state->heat = 0.0f;
    state->tool_condition = ABP_TOOL_MAX;
    state->alert = 0.0f;
    state->time_remaining = state->config.time_max;
    state->credits = 0.0f;

    for (c_idx = 0; c_idx < ABP_N_COMMODITIES; ++c_idx) {
        state->cargo[c_idx] = 0.0f;
    }

    abp_rng_seed(&state->rng, seed, 54u);
    abp_seed_world_stub(state);
    abp_pack_obs(state, NULL);
}

void abp_core_reset(AbpCoreState *state, uint64_t seed, float *obs_out) {
    abp_core_init(state, &state->config, seed);
    abp_pack_obs(state, obs_out);
}

void abp_core_step(AbpCoreState *state, uint8_t action, AbpCoreStepResult *out) {
    uint8_t terminated = 0u;
    uint8_t truncated = 0u;
    uint8_t invalid_action = 0u;
    uint16_t dt = 1u;
    float reward = 0.0f;

    if (action >= ABP_N_ACTIONS) {
        invalid_action = 1u;
        reward -= state->config.invalid_action_penalty;
        action = 6u;
    }

    if (action <= 5u) {
        int8_t next = state->neighbors[state->current_node][action];
        if (next >= 0) {
            state->current_node = (uint8_t)next;
            state->fuel -= 10.0f;
        } else {
            invalid_action = 1u;
            reward -= state->config.invalid_action_penalty;
        }
    } else if (action == 7u) {
        state->fuel -= 18.0f;
        state->alert += 10.0f;
    } else if (action == 68u) {
        terminated = 1u;
    }

    state->ticks_elapsed += dt;
    if (state->time_remaining > (float)dt) {
        state->time_remaining -= (float)dt;
    } else {
        state->time_remaining = 0.0f;
        truncated = 1u;
    }

    if (state->fuel < 0.0f) {
        state->fuel = 0.0f;
    }
    if (state->alert > ABP_ALERT_MAX) {
        state->alert = ABP_ALERT_MAX;
    }

    abp_pack_obs(state, out->obs);
    out->reward = reward;
    out->terminated = terminated;
    out->truncated = truncated;
    out->invalid_action = invalid_action;
    out->dt = dt;
}
