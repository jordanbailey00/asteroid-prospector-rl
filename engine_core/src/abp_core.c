
#include "abp_core.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define ABP_MKT_PRICE_BASE 244u
#define ABP_MKT_DPRICE_BASE 250u
#define ABP_MKT_INV_BASE 256u

#define ABP_PI_F 3.14159265358979323846f

#define ABP_CREDITS_CAP 10000000.0f

#define ABP_REPAIR_KITS_CAP 12u
#define ABP_STABILIZERS_CAP 12u
#define ABP_DECOYS_CAP 12u

#define ABP_TRAVEL_TIME_MAX 8.0f
#define ABP_TRAVEL_FUEL_COST_MAX 160.0f
#define ABP_INV_TRAVEL_TIME_MAX (1.0f / ABP_TRAVEL_TIME_MAX)
#define ABP_INV_TRAVEL_FUEL_COST_MAX (1.0f / ABP_TRAVEL_FUEL_COST_MAX)

#define ABP_PRICE_SCALE 100.0f
#define ABP_STATION_INVENTORY_NORM_CAP 500.0f
#define ABP_INV_STATION_INVENTORY_NORM_CAP (1.0f / ABP_STATION_INVENTORY_NORM_CAP)
#define ABP_INV_PRICE_SCALE (1.0f / ABP_PRICE_SCALE)
#define ABP_INV_MAX_NODE_INDEX (1.0f / (float)(ABP_MAX_NODES - 1u))

#define ABP_WIDE_SCAN_TIME 3u
#define ABP_FOCUSED_SCAN_TIME 2u
#define ABP_DEEP_SCAN_TIME 4u
#define ABP_THREAT_LISTEN_TIME 2u
#define ABP_STABILIZE_TIME 2u
#define ABP_REFINE_TIME 2u
#define ABP_COOLDOWN_TIME 2u
#define ABP_MAINT_TIME 2u
#define ABP_PATCH_TIME 2u
#define ABP_DOCK_TIME 1u
#define ABP_OVERHAUL_TIME 3u

#define ABP_WIDE_SCAN_FUEL 5.0f
#define ABP_FOCUSED_SCAN_FUEL 4.0f
#define ABP_DEEP_SCAN_FUEL 8.0f
#define ABP_REFINE_FUEL 4.0f
#define ABP_COOLDOWN_FUEL 2.0f
#define ABP_EMERGENCY_BURN_FUEL 18.0f

#define ABP_REFINE_HEAT 6.0f
#define ABP_COOLDOWN_AMOUNT 20.0f

#define ABP_EMERGENCY_BURN_ALERT 10.0f
#define ABP_WIDE_SCAN_ALERT 4.0f
#define ABP_FOCUSED_SCAN_ALERT 3.0f
#define ABP_DEEP_SCAN_ALERT 6.0f
#define ABP_REFINE_ALERT 3.0f
#define ABP_COOLDOWN_ALERT 1.0f
#define ABP_ALERT_DECAY_HOLD 3.0f
#define ABP_DOCK_ALERT_DROP 20.0f
#define ABP_JETTISON_ALERT_RELIEF 8.0f

#define ABP_HEAT_DISSIPATION_PER_TICK 2.5f
#define ABP_OVERHEAT_DAMAGE_PER_UNIT 1.25f

#define ABP_TOOL_REPAIR_AMOUNT 25.0f
#define ABP_HULL_PATCH_AMOUNT 20.0f

#define ABP_ESCAPE_BUFF_TICKS 4u
#define ABP_STABILIZE_BUFF_TICKS 6u

#define ABP_FRACTURE_DEPLETION_RATE 0.01f

#define ABP_HAZARD_DAMAGE_PER_TICK 0.7f
#define ABP_HAZARD_HEAT_PER_TICK 0.5f
#define ABP_HAZARD_ALERT_PER_TICK 0.8f

#define ABP_PIRATE_BIAS -4.0f
#define ABP_PIRATE_INTENSITY_W 3.0f
#define ABP_PIRATE_ALERT_W 2.2f
#define ABP_PIRATE_CARGO_W 0.8f
#define ABP_PIRATE_ESCAPE_W 2.8f

#define ABP_SLIPPAGE_K 0.25f
#define ABP_SLIPPAGE_ROOT 0.2f

#define ABP_INVENTORY_PRESSURE_K 0.04f
#define ABP_SALES_PRESSURE_K 0.05f
#define ABP_MARKET_NOISE_K 0.03f
#define ABP_SALES_DECAY_TAU 14.0f

#define ABP_BUY_FUEL_SMALL_QTY 120.0f
#define ABP_BUY_FUEL_MED_QTY 260.0f
#define ABP_BUY_FUEL_LARGE_QTY 480.0f

#define ABP_BUY_FUEL_SMALL_COST 60.0f
#define ABP_BUY_FUEL_MED_COST 120.0f
#define ABP_BUY_FUEL_LARGE_COST 210.0f
#define ABP_BUY_REPAIR_KIT_COST 150.0f
#define ABP_BUY_STABILIZER_COST 175.0f
#define ABP_BUY_DECOY_COST 110.0f

#define ABP_OVERHAUL_COST 280.0f

#define ABP_REWARD_ALPHA_EXTRACT 0.02f
#define ABP_REWARD_BETA_FUEL 0.10f
#define ABP_REWARD_GAMMA_TIME 0.001f
#define ABP_REWARD_DELTA_WEAR 0.05f
#define ABP_REWARD_EPSILON_HEAT 0.20f
#define ABP_REWARD_ZETA_DAMAGE 1.00f
#define ABP_REWARD_KAPPA_PIRATE 1.00f
#define ABP_REWARD_SCAN_COST 0.005f
#define ABP_REWARD_HEAT_SAFE_FRAC 0.70f
#define ABP_REWARD_STRANDED_PEN 50.0f
#define ABP_REWARD_DESTROYED_PEN 100.0f
#define ABP_REWARD_TERMINAL_BONUS_B 0.002f

static const float k_price_base[ABP_N_COMMODITIES] = {45.0f, 55.0f, 85.0f, 145.0f, 210.0f, 120.0f};
static const float k_inv_price_base[ABP_N_COMMODITIES] = {
    1.0f / 45.0f, 1.0f / 55.0f, 1.0f / 85.0f, 1.0f / 145.0f, 1.0f / 210.0f, 1.0f / 120.0f};
static const float k_price_min[ABP_N_COMMODITIES] = {12.0f, 15.0f, 20.0f, 50.0f, 80.0f, 30.0f};
static const float k_price_max[ABP_N_COMMODITIES] = {180.0f, 200.0f, 240.0f,
                                                     320.0f, 420.0f, 300.0f};

typedef struct AbpStepSnapshot {
    float credits_before;
    float fuel_before;
    float hull_before;
    float heat_before;
    float tool_before;
    float cargo_value_before;
    float value_lost_to_pirates_before;
} AbpStepSnapshot;

typedef struct AbpTravelResult {
    uint16_t dt;
    uint8_t invalid;
} AbpTravelResult;

static float abp_clampf(float value, float low, float high) {
    if (value < low) {
        return low;
    }
    if (value > high) {
        return high;
    }
    return value;
}

static float abp_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

static uint32_t abp_rng_u32_range(AbpCoreState *state, uint32_t low, uint32_t high_exclusive) {
    uint32_t span = 0;
    if (high_exclusive <= low) {
        return low;
    }

    span = high_exclusive - low;
    return low + (abp_rng_next_u32(&state->rng) % span);
}

static float abp_rng_uniform(AbpCoreState *state, float low, float high) {
    return low + (high - low) * abp_rng_next_f32(&state->rng);
}

static float abp_rng_exp_unit(AbpCoreState *state) {
    float u = abp_rng_next_f32(&state->rng);
    if (u < 1.0e-8f) {
        u = 1.0e-8f;
    }
    return -logf(u);
}

static float abp_rng_normal(AbpCoreState *state, float mean, float sigma) {
    float u1 = abp_rng_next_f32(&state->rng);
    float u2 = abp_rng_next_f32(&state->rng);
    float mag = 0.0f;
    float z0 = 0.0f;

    if (u1 < 1.0e-8f) {
        u1 = 1.0e-8f;
    }

    mag = sqrtf(-2.0f * logf(u1));
    z0 = mag * cosf(2.0f * ABP_PI_F * u2);
    return mean + sigma * z0;
}

static float abp_rng_lognormal(AbpCoreState *state, float mean, float sigma) {
    return expf(abp_rng_normal(state, mean, sigma));
}

static float abp_rng_beta_3_2(AbpCoreState *state) {
    float a = 0.0f;
    float b = 0.0f;
    float total = 0.0f;
    uint32_t i = 0;

    for (i = 0; i < 3u; ++i) {
        a += abp_rng_exp_unit(state);
    }
    for (i = 0; i < 2u; ++i) {
        b += abp_rng_exp_unit(state);
    }

    total = a + b;
    if (total <= 0.0f) {
        return 0.5f;
    }
    return a / total;
}

static void abp_rng_dirichlet_ones(AbpCoreState *state, float out[ABP_N_COMMODITIES]) {
    float sum = 0.0f;
    uint32_t i = 0;
    for (i = 0; i < ABP_N_COMMODITIES; ++i) {
        out[i] = abp_rng_exp_unit(state);
        sum += out[i];
    }

    if (sum <= 0.0f) {
        for (i = 0; i < ABP_N_COMMODITIES; ++i) {
            out[i] = 1.0f / (float)ABP_N_COMMODITIES;
        }
        return;
    }

    for (i = 0; i < ABP_N_COMMODITIES; ++i) {
        out[i] /= sum;
    }
}

static void abp_normalize_probs(const float *in, float *out, uint32_t n) {
    float sum = 0.0f;
    uint32_t i = 0;

    for (i = 0; i < n; ++i) {
        float v = in[i];
        if (v < 1.0e-8f) {
            v = 1.0e-8f;
        }
        out[i] = v;
        sum += v;
    }

    if (sum <= 0.0f) {
        float fill = 1.0f / (float)n;
        for (i = 0; i < n; ++i) {
            out[i] = fill;
        }
        return;
    }

    for (i = 0; i < n; ++i) {
        out[i] /= sum;
    }
}

static void abp_normalize_comp_est_6_to_obs(const float *in, float *out) {
    float sum = 0.0f;
    uint32_t i = 0;

    for (i = 0; i < ABP_N_COMMODITIES; ++i) {
        float value = in[i];
        if (value < 1.0e-8f) {
            value = 1.0e-8f;
        }
        out[i] = value;
        sum += value;
    }

    if (sum <= 0.0f) {
        float fill = 1.0f / (float)ABP_N_COMMODITIES;
        for (i = 0; i < ABP_N_COMMODITIES; ++i) {
            out[i] = fill;
        }
        return;
    }

    {
        float inv_sum = 1.0f / sum;
        for (i = 0; i < ABP_N_COMMODITIES; ++i) {
            out[i] *= inv_sum;
        }
    }
}

static void abp_recompute_steps_to_station(AbpCoreState *state) {
    uint8_t visited[ABP_MAX_NODES];
    uint8_t queue[ABP_MAX_NODES];
    uint8_t head = 0u;
    uint8_t tail = 0u;
    uint8_t node = 0u;

    memset(visited, 0, sizeof(visited));
    for (node = 0; node < ABP_MAX_NODES; ++node) {
        state->steps_to_station[node] = (uint8_t)(ABP_MAX_NODES - 1u);
    }

    if (state->node_count == 0u) {
        return;
    }

    visited[0] = 1u;
    state->steps_to_station[0] = 0u;
    queue[tail++] = 0u;

    while (head < tail) {
        uint8_t cur = queue[head++];
        uint8_t cur_dist = state->steps_to_station[cur];
        uint8_t slot = 0u;

        for (slot = 0u; slot < ABP_MAX_NEIGHBORS; ++slot) {
            int neighbor = state->neighbors[cur][slot];
            if (neighbor < 0 || neighbor >= (int)state->node_count) {
                continue;
            }
            if (visited[neighbor] > 0u) {
                continue;
            }

            visited[neighbor] = 1u;
            if (cur_dist < (uint8_t)(ABP_MAX_NODES - 1u)) {
                state->steps_to_station[neighbor] = (uint8_t)(cur_dist + 1u);
            } else {
                state->steps_to_station[neighbor] = (uint8_t)(ABP_MAX_NODES - 1u);
            }
            if (tail < (uint8_t)ABP_MAX_NODES) {
                queue[tail++] = (uint8_t)neighbor;
            }
        }
    }
}

static float abp_cargo_sum(const AbpCoreState *state) {
    float total = 0.0f;
    uint32_t i = 0;
    for (i = 0; i < ABP_N_COMMODITIES; ++i) {
        total += state->cargo[i];
    }
    return total;
}

static int abp_is_at_station(const AbpCoreState *state) {
    return state->node_type[state->current_node] == ABP_NODE_STATION;
}

static int abp_selected_asteroid_valid(const AbpCoreState *state) {
    int a_idx = state->selected_asteroid;
    if (a_idx < 0 || a_idx >= ABP_MAX_ASTEROIDS) {
        return 0;
    }
    if (state->ast_valid[state->current_node][a_idx] == 0u) {
        return 0;
    }
    if (state->depletion[state->current_node][a_idx] >= 1.0f) {
        return 0;
    }
    return 1;
}

static float abp_est_cargo_value(const AbpCoreState *state) {
    float value = 0.0f;
    uint32_t c_idx = 0;
    for (c_idx = 0; c_idx < ABP_N_COMMODITIES; ++c_idx) {
        value += state->cargo[c_idx] * state->market_price[c_idx];
    }
    return value;
}

static int abp_steps_to_station(const AbpCoreState *state) {
    if (state->current_node >= state->node_count) {
        return ABP_MAX_NODES - 1;
    }
    return (int)state->steps_to_station[state->current_node];
}

static int abp_edge_exists(const AbpCoreState *state, int u, int v) {
    int slot = 0;
    for (slot = 0; slot < ABP_MAX_NEIGHBORS; ++slot) {
        if (state->neighbors[u][slot] == v) {
            return 1;
        }
    }
    return 0;
}

static int abp_first_free_slot(const AbpCoreState *state, int node) {
    int slot = 0;
    for (slot = 0; slot < ABP_MAX_NEIGHBORS; ++slot) {
        if (state->neighbors[node][slot] < 0) {
            return slot;
        }
    }
    return -1;
}

static void abp_add_edge(AbpCoreState *state, int u, int v) {
    int u_slot = 0;
    int v_slot = 0;
    uint32_t t_time = 0;
    float fuel_cost = 0.0f;
    float threat = 0.0f;

    if (u >= (int)state->node_count || v >= (int)state->node_count) {
        return;
    }
    if (abp_edge_exists(state, u, v)) {
        return;
    }

    u_slot = abp_first_free_slot(state, u);
    v_slot = abp_first_free_slot(state, v);
    if (u_slot < 0 || v_slot < 0) {
        return;
    }

    t_time = abp_rng_u32_range(state, 1u, (uint32_t)ABP_TRAVEL_TIME_MAX + 1u);
    fuel_cost = abp_rng_uniform(state, 20.0f, ABP_TRAVEL_FUEL_COST_MAX * 0.7f);

    threat = 0.5f * (state->node_hazard[u] + state->node_hazard[v]) +
             0.5f * (state->node_pirate[u] + state->node_pirate[v]) +
             abp_rng_normal(state, 0.0f, 0.05f);
    threat = abp_clampf(threat, 0.0f, 1.0f);

    state->neighbors[u][u_slot] = (int8_t)v;
    state->neighbors[v][v_slot] = (int8_t)u;

    state->edge_travel_time[u][u_slot] = (uint8_t)t_time;
    state->edge_travel_time[v][v_slot] = (uint8_t)t_time;

    state->edge_fuel_cost[u][u_slot] = fuel_cost;
    state->edge_fuel_cost[v][v_slot] = fuel_cost;

    state->edge_threat_true[u][u_slot] = threat;
    state->edge_threat_true[v][v_slot] = threat;
    state->edge_threat_est[u][u_slot] = 0.5f;
    state->edge_threat_est[v][v_slot] = 0.5f;
}
static void abp_generate_asteroids(AbpCoreState *state) {
    uint32_t node = 0;

    memset(state->ast_valid, 0, sizeof(state->ast_valid));
    memset(state->true_comp, 0, sizeof(state->true_comp));
    memset(state->richness, 0, sizeof(state->richness));
    memset(state->stability_true, 0, sizeof(state->stability_true));
    memset(state->noise_profile, 0, sizeof(state->noise_profile));

    memset(state->comp_est, 0, sizeof(state->comp_est));
    memset(state->stability_est, 0, sizeof(state->stability_est));
    memset(state->scan_conf, 0, sizeof(state->scan_conf));
    memset(state->depletion, 0, sizeof(state->depletion));

    for (node = 0; node < state->node_count; ++node) {
        uint32_t n_ast = 0;
        uint32_t a_idx = 0;
        if (state->node_type[node] == ABP_NODE_STATION) {
            continue;
        }

        n_ast = abp_rng_u32_range(state, 5u, ABP_MAX_ASTEROIDS + 1u);
        for (a_idx = 0; a_idx < n_ast; ++a_idx) {
            float dir[ABP_N_COMMODITIES];
            float dir_est[ABP_N_COMMODITIES];
            uint32_t c_idx = 0;

            state->ast_valid[node][a_idx] = 1u;

            abp_rng_dirichlet_ones(state, dir);
            for (c_idx = 0; c_idx < ABP_N_COMMODITIES; ++c_idx) {
                state->true_comp[node][a_idx][c_idx] = dir[c_idx];
            }

            state->richness[node][a_idx] =
                abp_clampf(abp_rng_lognormal(state, -0.2f, 0.65f), 0.2f, 4.0f);
            state->stability_true[node][a_idx] = abp_rng_beta_3_2(state);
            state->noise_profile[node][a_idx] = abp_rng_uniform(state, 0.04f, 0.22f);

            abp_rng_dirichlet_ones(state, dir_est);
            for (c_idx = 0; c_idx < ABP_N_COMMODITIES; ++c_idx) {
                state->comp_est[node][a_idx][c_idx] = dir_est[c_idx];
            }
            state->stability_est[node][a_idx] = 0.5f;
            state->scan_conf[node][a_idx] = 0.1f;
            state->depletion[node][a_idx] = 0.0f;
        }
    }
}

static void abp_generate_market(AbpCoreState *state) {
    uint32_t c_idx = 0;

    memset(state->recent_sales, 0, sizeof(state->recent_sales));

    for (c_idx = 0; c_idx < ABP_N_COMMODITIES; ++c_idx) {
        float phase = 0.0f;
        float period = 0.0f;
        float amp_factor = 0.0f;
        float cycle = 0.0f;
        float price = 0.0f;

        state->station_inventory[c_idx] = abp_rng_uniform(state, 20.0f, 120.0f);

        phase = abp_rng_uniform(state, 0.0f, 2.0f * ABP_PI_F);
        period = abp_rng_uniform(state, 180.0f, 380.0f);
        amp_factor = abp_rng_uniform(state, 0.10f, 0.30f);

        state->price_phase[c_idx] = phase;
        state->price_period[c_idx] = period;
        state->price_amp[c_idx] = k_price_base[c_idx] * amp_factor;

        cycle = state->price_amp[c_idx] * sinf(state->price_phase[c_idx]);
        price = k_price_base[c_idx] + cycle;
        price = abp_clampf(price, k_price_min[c_idx], k_price_max[c_idx]);

        state->market_price[c_idx] = price;
        state->market_prev_price[c_idx] = price;
    }
}

static void abp_generate_world(AbpCoreState *state) {
    uint32_t node = 0;

    state->node_count = (uint8_t)abp_rng_u32_range(state, 8u, ABP_MAX_NODES + 1u);
    state->current_node = 0u;

    for (node = 0; node < ABP_MAX_NODES; ++node) {
        uint32_t slot = 0;
        state->node_type[node] = ABP_NODE_CLUSTER;
        state->node_hazard[node] = 0.0f;
        state->node_pirate[node] = 0.0f;
        state->steps_to_station[node] = (uint8_t)(ABP_MAX_NODES - 1u);
        for (slot = 0; slot < ABP_MAX_NEIGHBORS; ++slot) {
            state->neighbors[node][slot] = -1;
            state->edge_travel_time[node][slot] = 1u;
            state->edge_fuel_cost[node][slot] = 0.0f;
            state->edge_threat_true[node][slot] = 0.0f;
            state->edge_threat_est[node][slot] = 0.5f;
        }
    }

    state->node_type[0] = ABP_NODE_STATION;

    for (node = 1; node < state->node_count; ++node) {
        state->node_type[node] =
            (abp_rng_next_f32(&state->rng) < 0.25f) ? ABP_NODE_HAZARD : ABP_NODE_CLUSTER;
        state->node_hazard[node] = abp_rng_uniform(state, 0.05f, 0.35f);
        state->node_pirate[node] = abp_rng_uniform(state, 0.05f, 0.30f);
        if (state->node_type[node] == ABP_NODE_HAZARD) {
            state->node_hazard[node] += 0.25f;
            state->node_pirate[node] += 0.12f;
        }
    }

    for (node = 1; node < state->node_count; ++node) {
        uint32_t parent = abp_rng_u32_range(state, 0u, node);
        abp_add_edge(state, (int)node, (int)parent);
    }

    for (node = 0; node < state->node_count; ++node) {
        uint32_t u = abp_rng_u32_range(state, 0u, state->node_count);
        uint32_t v = abp_rng_u32_range(state, 0u, state->node_count);
        if (u == v) {
            continue;
        }
        abp_add_edge(state, (int)u, (int)v);
    }

    abp_recompute_steps_to_station(state);

    abp_generate_asteroids(state);
    abp_generate_market(state);
}

static void abp_passive_heat_dissipation(AbpCoreState *state, uint16_t dt) {
    state->heat = fmaxf(0.0f, state->heat - ABP_HEAT_DISSIPATION_PER_TICK * (float)dt);
}

static void abp_apply_hold(AbpCoreState *state) {
    state->alert = fmaxf(0.0f, state->alert - ABP_ALERT_DECAY_HOLD);
    abp_passive_heat_dissipation(state, 1u);
}

static void abp_apply_emergency_burn(AbpCoreState *state) {
    state->fuel -= ABP_EMERGENCY_BURN_FUEL;
    state->alert += ABP_EMERGENCY_BURN_ALERT;
    if (state->escape_buff_ticks < ABP_ESCAPE_BUFF_TICKS) {
        state->escape_buff_ticks = ABP_ESCAPE_BUFF_TICKS;
    }
}

static void abp_maybe_pirate_encounter(AbpCoreState *state, uint16_t dt, float intensity) {
    float pirate_intensity = 0.0f;
    float cargo_value_before = 0.0f;
    float logit = 0.0f;
    float base_prob = 0.0f;
    float p_encounter = 0.0f;

    if (abp_is_at_station(state)) {
        return;
    }

    pirate_intensity = intensity;
    if (pirate_intensity < 0.0f) {
        pirate_intensity = state->node_pirate[state->current_node];
    }

    cargo_value_before = abp_est_cargo_value(state);

    logit = ABP_PIRATE_BIAS + ABP_PIRATE_INTENSITY_W * pirate_intensity +
            ABP_PIRATE_ALERT_W * abp_clampf(state->alert / ABP_ALERT_MAX, 0.0f, 1.0f) +
            ABP_PIRATE_CARGO_W * log1pf(cargo_value_before / ABP_CREDIT_SCALE) -
            ABP_PIRATE_ESCAPE_W * (state->escape_buff_ticks > 0u ? 1.0f : 0.0f);

    base_prob = abp_sigmoid(logit);
    p_encounter = 1.0f - powf(1.0f - base_prob, (float)(dt == 0u ? 1u : dt));

    if (abp_rng_next_f32(&state->rng) >= p_encounter) {
        return;
    }

    state->pirate_encounters += 1u;

    {
        float loss_frac = abp_rng_uniform(state, 0.08f, 0.20f);
        float cargo_value_after = 0.0f;
        uint32_t c_idx = 0;

        if (state->decoys > 0u && abp_rng_next_f32(&state->rng) < 0.6f) {
            state->decoys -= 1u;
            loss_frac *= 0.3f;
        }

        for (c_idx = 0; c_idx < ABP_N_COMMODITIES; ++c_idx) {
            state->cargo[c_idx] *= (1.0f - loss_frac);
        }

        cargo_value_after = abp_est_cargo_value(state);
        if (cargo_value_before > cargo_value_after) {
            state->value_lost_to_pirates += cargo_value_before - cargo_value_after;
        }

        state->hull -= abp_rng_uniform(state, 1.0f, 4.0f);
        state->alert += 8.0f;
    }
}

static void abp_apply_edge_hazards_and_pirates(AbpCoreState *state, uint16_t dt,
                                               float edge_threat) {
    float hazard_dmg = 0.0f;
    if (dt == 0u) {
        return;
    }

    hazard_dmg = (float)dt * edge_threat * ABP_HAZARD_DAMAGE_PER_TICK;
    hazard_dmg *= abp_rng_uniform(state, 0.85f, 1.15f);
    state->hull -= hazard_dmg;

    state->heat += (float)dt * edge_threat * ABP_HAZARD_HEAT_PER_TICK;
    state->alert += (float)dt * edge_threat * ABP_HAZARD_ALERT_PER_TICK;

    abp_maybe_pirate_encounter(state, dt, edge_threat);
}

static AbpTravelResult abp_apply_travel(AbpCoreState *state, uint8_t slot) {
    AbpTravelResult result;
    int neighbor = state->neighbors[state->current_node][slot];

    result.dt = 1u;
    result.invalid = 0u;

    if (neighbor < 0) {
        result.invalid = 1u;
        return result;
    }

    result.dt = state->edge_travel_time[state->current_node][slot];
    if (result.dt == 0u) {
        result.dt = 1u;
    }

    {
        float mass_factor = 1.0f + 0.5f * (abp_cargo_sum(state) / ABP_CARGO_MAX);
        float fuel_cost = state->edge_fuel_cost[state->current_node][slot] * mass_factor;
        float threat = state->edge_threat_true[state->current_node][slot];

        state->fuel -= fuel_cost;
        state->current_node = (uint8_t)neighbor;
        state->selected_asteroid = -1;

        abp_apply_edge_hazards_and_pirates(state, result.dt, threat);
    }

    return result;
}

static void abp_update_asteroid_estimates(AbpCoreState *state, uint8_t asteroid, uint8_t mode) {
    float blend = 0.0f;
    float conf_gain = 0.0f;
    float noise_mult = 0.0f;
    uint8_t node = state->current_node;
    float base_noise = 0.0f;
    float conf = 0.0f;
    float sigma = 0.0f;
    float noisy_truth_raw[ABP_N_COMMODITIES];
    float noisy_truth[ABP_N_COMMODITIES];
    float mixed[ABP_N_COMMODITIES];
    float mixed_norm[ABP_N_COMMODITIES];
    float stable_truth = 0.0f;
    float stable_noisy = 0.0f;
    float stable_est = 0.0f;
    uint32_t c_idx = 0;

    if (state->ast_valid[node][asteroid] == 0u) {
        return;
    }

    if (mode == 0u) {
        blend = 0.22f;
        conf_gain = 0.10f;
        noise_mult = 1.35f;
    } else if (mode == 1u) {
        blend = 0.42f;
        conf_gain = 0.20f;
        noise_mult = 1.0f;
    } else {
        blend = 0.80f;
        conf_gain = 0.45f;
        noise_mult = 0.55f;
    }

    base_noise = state->noise_profile[node][asteroid];
    conf = state->scan_conf[node][asteroid];
    sigma = base_noise * (1.0f - conf + 0.1f) * noise_mult;

    for (c_idx = 0; c_idx < ABP_N_COMMODITIES; ++c_idx) {
        noisy_truth_raw[c_idx] =
            state->true_comp[node][asteroid][c_idx] + abp_rng_normal(state, 0.0f, sigma);
    }
    abp_normalize_probs(noisy_truth_raw, noisy_truth, ABP_N_COMMODITIES);

    for (c_idx = 0; c_idx < ABP_N_COMMODITIES; ++c_idx) {
        mixed[c_idx] =
            (1.0f - blend) * state->comp_est[node][asteroid][c_idx] + blend * noisy_truth[c_idx];
    }
    abp_normalize_probs(mixed, mixed_norm, ABP_N_COMMODITIES);

    for (c_idx = 0; c_idx < ABP_N_COMMODITIES; ++c_idx) {
        state->comp_est[node][asteroid][c_idx] = mixed_norm[c_idx];
    }

    stable_truth = state->stability_true[node][asteroid];
    stable_noisy = abp_clampf(stable_truth + abp_rng_normal(state, 0.0f, sigma), 0.0f, 1.0f);
    stable_est = (1.0f - blend) * state->stability_est[node][asteroid] + blend * stable_noisy;
    state->stability_est[node][asteroid] = abp_clampf(stable_est, 0.0f, 1.0f);

    state->scan_conf[node][asteroid] =
        abp_clampf(state->scan_conf[node][asteroid] + conf_gain, 0.0f, 1.0f);
}

static void abp_update_cluster_priors_with_noise(AbpCoreState *state) {
    uint8_t node = state->current_node;
    uint8_t a_idx = 0;
    for (a_idx = 0; a_idx < ABP_MAX_ASTEROIDS; ++a_idx) {
        if (state->ast_valid[node][a_idx] > 0u) {
            abp_update_asteroid_estimates(state, a_idx, 0u);
        }
    }
}

static void abp_update_neighbor_threat_estimates(AbpCoreState *state) {
    uint8_t slot = 0;
    for (slot = 0; slot < ABP_MAX_NEIGHBORS; ++slot) {
        int neighbor = state->neighbors[state->current_node][slot];
        if (neighbor < 0) {
            continue;
        }

        {
            float truth = state->edge_threat_true[state->current_node][slot];
            float est = state->edge_threat_est[state->current_node][slot];
            float noisy = abp_clampf(truth + abp_rng_normal(state, 0.0f, 0.08f), 0.0f, 1.0f);
            state->edge_threat_est[state->current_node][slot] = 0.25f * est + 0.75f * noisy;
        }
    }
}

static int abp_select_asteroid(AbpCoreState *state, int asteroid) {
    if (asteroid < 0 || asteroid >= ABP_MAX_ASTEROIDS) {
        return 0;
    }
    if (state->ast_valid[state->current_node][asteroid] == 0u) {
        return 0;
    }
    if (state->depletion[state->current_node][asteroid] >= 1.0f) {
        return 0;
    }
    state->selected_asteroid = (int8_t)asteroid;
    return 1;
}
static void abp_mine_selected(AbpCoreState *state, uint8_t action) {
    float mode_mult = 1.0f;
    float heat_gain = 0.0f;
    float wear_gain = 0.0f;
    float alert_gain = 0.0f;
    float sigma = 0.0f;
    float fracture_bias = 0.0f;
    uint8_t node = state->current_node;
    uint8_t a_idx = (uint8_t)state->selected_asteroid;

    float richness = 0.0f;
    float depletion = 0.0f;
    float base = 0.0f;
    float tool_frac = 0.0f;
    float heat_frac = 0.0f;
    float eff_tool = 0.0f;
    float eff_heat = 0.0f;
    float noise = 0.0f;
    float extracted[ABP_N_COMMODITIES];
    float available_capacity = 0.0f;
    float total_extracted = 0.0f;
    float logit = 0.0f;
    uint32_t c_idx = 0;

    if (action == 28u) {
        mode_mult = 0.80f;
        heat_gain = 2.0f;
        wear_gain = 0.8f;
        alert_gain = 1.2f;
        sigma = 0.05f;
        fracture_bias = -0.7f;
    } else if (action == 29u) {
        mode_mult = 1.15f;
        heat_gain = 4.0f;
        wear_gain = 1.6f;
        alert_gain = 2.2f;
        sigma = 0.10f;
        fracture_bias = 0.0f;
    } else {
        mode_mult = 1.55f;
        heat_gain = 7.0f;
        wear_gain = 2.8f;
        alert_gain = 4.0f;
        sigma = 0.16f;
        fracture_bias = 0.8f;
    }

    richness = state->richness[node][a_idx];
    depletion = state->depletion[node][a_idx];
    base = richness * fmaxf(0.0f, 1.0f - depletion);

    tool_frac = abp_clampf(state->tool_condition / ABP_TOOL_MAX, 0.0f, 1.0f);
    heat_frac = abp_clampf(state->heat / ABP_HEAT_MAX, 0.0f, 2.0f);

    eff_tool = 0.4f + 0.6f * tool_frac;
    if (heat_frac <= 0.7f) {
        eff_heat = 1.0f;
    } else {
        eff_heat = fmaxf(0.1f, 1.0f - (heat_frac - 0.7f) / 0.3f);
    }

    noise = expf(abp_rng_normal(state, 0.0f, sigma));
    for (c_idx = 0; c_idx < ABP_N_COMMODITIES; ++c_idx) {
        extracted[c_idx] =
            base * eff_tool * eff_heat * mode_mult * noise * state->true_comp[node][a_idx][c_idx];
        total_extracted += extracted[c_idx];
    }

    available_capacity = fmaxf(0.0f, ABP_CARGO_MAX - abp_cargo_sum(state));
    if (total_extracted > available_capacity && total_extracted > 0.0f) {
        float scale = available_capacity / total_extracted;
        total_extracted = available_capacity;
        for (c_idx = 0; c_idx < ABP_N_COMMODITIES; ++c_idx) {
            extracted[c_idx] *= scale;
        }
    }

    for (c_idx = 0; c_idx < ABP_N_COMMODITIES; ++c_idx) {
        state->cargo[c_idx] += extracted[c_idx];
    }
    state->heat += heat_gain;
    state->tool_condition -= wear_gain;
    state->alert += alert_gain;

    state->depletion[node][a_idx] = abp_clampf(
        state->depletion[node][a_idx] + ABP_FRACTURE_DEPLETION_RATE * total_extracted, 0.0f, 1.0f);

    state->mining_ticks += 1u;

    logit = -3.1f + fracture_bias + 2.5f * (1.0f - state->stability_true[node][a_idx]) +
            2.2f * fmaxf(0.0f, heat_frac - 0.7f) + 1.5f * (1.0f - tool_frac) -
            ((state->stabilize_buff_ticks[a_idx] > 0u) ? 1.1f : 0.0f);

    if (abp_rng_next_f32(&state->rng) < abp_sigmoid(logit)) {
        float severity = abp_rng_uniform(state, 0.5f, 1.0f);
        state->hull -= 12.0f * severity;
        state->depletion[node][a_idx] = 1.0f;
        state->node_hazard[node] = abp_clampf(state->node_hazard[node] + 0.1f, 0.0f, 1.0f);
    }
}

static void abp_refine_some_cargo(AbpCoreState *state) {
    float low_value = state->cargo[0] + state->cargo[1];
    float refine_input = 0.0f;
    float total_low = 0.0f;
    float take_ratio = 0.0f;
    float output = 0.0f;

    if (low_value <= 0.0f) {
        return;
    }

    refine_input = 0.15f * low_value;
    total_low = state->cargo[0] + state->cargo[1];
    if (total_low <= 0.0f) {
        return;
    }

    take_ratio = fminf(1.0f, refine_input / total_low);
    state->cargo[0] *= 1.0f - take_ratio;
    state->cargo[1] *= 1.0f - take_ratio;

    output = 0.65f * refine_input;
    state->cargo[4] += output;
}

static float abp_slippage(float qty, float inventory) {
    float ratio = 0.0f;
    float raw = 0.0f;

    if (qty <= 0.0f) {
        return 0.0f;
    }

    ratio = qty / fmaxf(1.0f, inventory + qty);
    raw = ABP_SLIPPAGE_K * ratio + ABP_SLIPPAGE_ROOT * sqrtf(ratio);
    return abp_clampf(raw, 0.0f, 0.70f);
}

static void abp_sell_action(AbpCoreState *state, uint8_t action) {
    uint8_t c_idx = (uint8_t)((action - 43u) / 3u);
    uint8_t bucket = (uint8_t)((action - 43u) % 3u);
    float frac = 1.0f;
    float qty = 0.0f;
    float slippage = 0.0f;
    float effective_price = 0.0f;

    if (bucket == 0u) {
        frac = 0.25f;
    } else if (bucket == 1u) {
        frac = 0.50f;
    } else {
        frac = 1.0f;
    }

    qty = state->cargo[c_idx] * frac;
    if (qty <= 0.0f) {
        return;
    }

    slippage = abp_slippage(qty, state->station_inventory[c_idx]);
    effective_price = state->market_price[c_idx] * (1.0f - slippage);

    state->credits += qty * effective_price;
    state->cargo[c_idx] = fmaxf(0.0f, state->cargo[c_idx] - qty);
    state->station_inventory[c_idx] += qty;
    state->recent_sales[c_idx] += qty;
}

static int abp_buy_fuel(AbpCoreState *state, float qty, float cost) {
    if (state->credits < cost) {
        return 0;
    }
    state->credits -= cost;
    state->total_spend += cost;
    state->fuel = fminf(ABP_FUEL_MAX, state->fuel + qty);
    return 1;
}

static int abp_buy_supply(AbpCoreState *state, uint8_t kind) {
    if (kind == 0u) {
        if (state->credits < ABP_BUY_REPAIR_KIT_COST || state->repair_kits >= ABP_REPAIR_KITS_CAP) {
            return 0;
        }
        state->credits -= ABP_BUY_REPAIR_KIT_COST;
        state->total_spend += ABP_BUY_REPAIR_KIT_COST;
        state->repair_kits += 1u;
        return 1;
    }

    if (kind == 1u) {
        if (state->credits < ABP_BUY_STABILIZER_COST || state->stabilizers >= ABP_STABILIZERS_CAP) {
            return 0;
        }
        state->credits -= ABP_BUY_STABILIZER_COST;
        state->total_spend += ABP_BUY_STABILIZER_COST;
        state->stabilizers += 1u;
        return 1;
    }

    if (state->credits < ABP_BUY_DECOY_COST || state->decoys >= ABP_DECOYS_CAP) {
        return 0;
    }
    state->credits -= ABP_BUY_DECOY_COST;
    state->total_spend += ABP_BUY_DECOY_COST;
    state->decoys += 1u;
    return 1;
}

static int abp_purchase_station_item(AbpCoreState *state, uint8_t action) {
    if (action == 61u) {
        return abp_buy_fuel(state, ABP_BUY_FUEL_SMALL_QTY, ABP_BUY_FUEL_SMALL_COST);
    }
    if (action == 62u) {
        return abp_buy_fuel(state, ABP_BUY_FUEL_MED_QTY, ABP_BUY_FUEL_MED_COST);
    }
    if (action == 63u) {
        return abp_buy_fuel(state, ABP_BUY_FUEL_LARGE_QTY, ABP_BUY_FUEL_LARGE_COST);
    }
    if (action == 64u) {
        return abp_buy_supply(state, 0u);
    }
    if (action == 65u) {
        return abp_buy_supply(state, 1u);
    }
    if (action == 66u) {
        return abp_buy_supply(state, 2u);
    }
    return 0;
}

static void abp_apply_node_hazards(AbpCoreState *state, uint16_t dt) {
    float hazard = state->node_hazard[state->current_node];
    float hull_damage = 0.0f;
    float heat_gain = 0.0f;
    float alert_gain = 0.0f;

    if (hazard <= 0.0f) {
        return;
    }

    hull_damage =
        (float)dt * hazard * ABP_HAZARD_DAMAGE_PER_TICK * abp_rng_uniform(state, 0.8f, 1.2f);
    heat_gain = (float)dt * hazard * ABP_HAZARD_HEAT_PER_TICK;
    alert_gain = (float)dt * hazard * ABP_HAZARD_ALERT_PER_TICK;

    state->hull -= hull_damage;
    state->heat += heat_gain;
    state->alert += alert_gain;
}

static void abp_update_market(AbpCoreState *state, uint16_t dt) {
    uint32_t c_idx = 0;
    float t = (float)(state->ticks_elapsed + dt);

    for (c_idx = 0; c_idx < ABP_N_COMMODITIES; ++c_idx) {
        state->market_prev_price[c_idx] = state->market_price[c_idx];
    }

    for (c_idx = 0; c_idx < ABP_N_COMMODITIES; ++c_idx) {
        float cycles = 0.0f;
        float inv_pressure = 0.0f;
        float sale_pressure = 0.0f;
        float noise_std = 0.0f;
        float noise = 0.0f;
        float new_price = 0.0f;

        cycles = state->price_amp[c_idx] * sinf(2.0f * ABP_PI_F * (t / state->price_period[c_idx]) +
                                                state->price_phase[c_idx]);
        inv_pressure = ABP_INVENTORY_PRESSURE_K * state->station_inventory[c_idx];
        sale_pressure = ABP_SALES_PRESSURE_K * state->recent_sales[c_idx];

        noise_std = ABP_MARKET_NOISE_K * k_price_base[c_idx] * sqrtf((float)(dt > 0u ? dt : 1u));
        noise = abp_rng_normal(state, 0.0f, noise_std);

        new_price = k_price_base[c_idx] + cycles - inv_pressure - sale_pressure + noise;
        state->market_price[c_idx] = abp_clampf(new_price, k_price_min[c_idx], k_price_max[c_idx]);
    }

    {
        float decay = expf(-(float)dt / ABP_SALES_DECAY_TAU);
        for (c_idx = 0; c_idx < ABP_N_COMMODITIES; ++c_idx) {
            state->recent_sales[c_idx] *= decay;
            state->station_inventory[c_idx] = fmaxf(state->station_inventory[c_idx] * 0.998f, 0.0f);
        }
    }
}

static void abp_clamp_state(AbpCoreState *state) {
    uint32_t c_idx = 0;
    float total_cargo = 0.0f;

    state->fuel = abp_clampf(state->fuel, 0.0f, ABP_FUEL_MAX);
    state->hull = abp_clampf(state->hull, 0.0f, ABP_HULL_MAX);
    state->heat = abp_clampf(state->heat, 0.0f, ABP_HEAT_MAX);
    state->tool_condition = abp_clampf(state->tool_condition, 0.0f, ABP_TOOL_MAX);
    state->alert = abp_clampf(state->alert, 0.0f, ABP_ALERT_MAX);
    state->time_remaining = abp_clampf(state->time_remaining, 0.0f, state->config.time_max);

    for (c_idx = 0; c_idx < ABP_N_COMMODITIES; ++c_idx) {
        state->cargo[c_idx] = abp_clampf(state->cargo[c_idx], 0.0f, ABP_CARGO_MAX);
        total_cargo += state->cargo[c_idx];
    }

    if (total_cargo > ABP_CARGO_MAX && total_cargo > 0.0f) {
        float scale = ABP_CARGO_MAX / total_cargo;
        for (c_idx = 0; c_idx < ABP_N_COMMODITIES; ++c_idx) {
            state->cargo[c_idx] *= scale;
        }
    }
}

static void abp_track_cargo_utilization(AbpCoreState *state, uint16_t dt) {
    float frac = abp_clampf(abp_cargo_sum(state) / ABP_CARGO_MAX, 0.0f, 1.0f);
    state->cargo_util_sum += frac * (float)dt;
    state->cargo_util_count += (float)dt;
}

static void abp_apply_global_dynamics(AbpCoreState *state, uint16_t dt) {
    uint32_t a_idx = 0;

    state->time_remaining -= (float)dt;

    abp_passive_heat_dissipation(state, dt);

    if (state->escape_buff_ticks > 0u) {
        state->escape_buff_ticks =
            (state->escape_buff_ticks > dt) ? (uint8_t)(state->escape_buff_ticks - dt) : 0u;
    }

    for (a_idx = 0; a_idx < ABP_MAX_ASTEROIDS; ++a_idx) {
        if (state->stabilize_buff_ticks[a_idx] > 0u) {
            state->stabilize_buff_ticks[a_idx] =
                (state->stabilize_buff_ticks[a_idx] > dt)
                    ? (uint8_t)(state->stabilize_buff_ticks[a_idx] - dt)
                    : 0u;
        }
    }

    if (state->heat > ABP_HEAT_MAX) {
        float overflow = state->heat - ABP_HEAT_MAX;
        state->hull -= ABP_OVERHEAT_DAMAGE_PER_UNIT * overflow;
        state->heat = ABP_HEAT_MAX;
        state->overheat_ticks += dt;
    }

    if (!abp_is_at_station(state)) {
        abp_apply_node_hazards(state, dt);
        abp_maybe_pirate_encounter(state, dt, -1.0f);
    }

    abp_update_market(state, dt);
    abp_clamp_state(state);
    abp_track_cargo_utilization(state, dt);
}

static float abp_compute_reward(const AbpCoreState *state, const AbpStepSnapshot *snapshot,
                                float cargo_value_after, uint8_t action, uint16_t dt,
                                uint8_t invalid, uint8_t destroyed, uint8_t stranded,
                                uint8_t done) {
    float delta_credits = state->credits - snapshot->credits_before;
    float r_sell = delta_credits / ABP_CREDIT_SCALE;

    float delta_cargo_value = fmaxf(0.0f, cargo_value_after - snapshot->cargo_value_before);
    float r_extract = ABP_REWARD_ALPHA_EXTRACT * (delta_cargo_value / ABP_CREDIT_SCALE);

    float r_fuel =
        -ABP_REWARD_BETA_FUEL * fmaxf(0.0f, snapshot->fuel_before - state->fuel) / 100.0f;
    float r_time = -ABP_REWARD_GAMMA_TIME * (float)dt;
    float r_wear =
        -ABP_REWARD_DELTA_WEAR * fmaxf(0.0f, snapshot->tool_before - state->tool_condition) / 10.0f;
    float r_damage =
        -ABP_REWARD_ZETA_DAMAGE * fmaxf(0.0f, snapshot->hull_before - state->hull) / 10.0f;

    float heat_safe = ABP_REWARD_HEAT_SAFE_FRAC * ABP_HEAT_MAX;
    float heat_excess = fmaxf(0.0f, state->heat - heat_safe);
    float heat_term = heat_excess / ABP_HEAT_MAX;
    float r_heat = -ABP_REWARD_EPSILON_HEAT * heat_term * heat_term;

    float r_scan =
        ((action == 8u) || (action == 9u) || (action == 10u)) ? -ABP_REWARD_SCAN_COST : 0.0f;
    float r_invalid = invalid ? -state->config.invalid_action_penalty : 0.0f;

    float delta_pirate_loss =
        fmaxf(0.0f, state->value_lost_to_pirates - snapshot->value_lost_to_pirates_before);
    float r_pirate = -ABP_REWARD_KAPPA_PIRATE * (delta_pirate_loss / ABP_CREDIT_SCALE);

    float r_terminal = 0.0f;
    if (stranded) {
        r_terminal -= ABP_REWARD_STRANDED_PEN;
    }
    if (destroyed) {
        r_terminal -= ABP_REWARD_DESTROYED_PEN;
    }
    if (done && !destroyed && !stranded) {
        r_terminal += ABP_REWARD_TERMINAL_BONUS_B * (state->credits / ABP_CREDIT_SCALE);
    }

    return r_sell + r_extract + r_fuel + r_time + r_wear + r_heat + r_damage + r_scan + r_invalid +
           r_pirate + r_terminal;
}

static void abp_fill_step_metrics(const AbpCoreState *state, AbpCoreStepResult *out,
                                  uint8_t destroyed, uint8_t stranded) {
    float net_profit = state->credits - state->total_spend;
    float profit_per_tick =
        net_profit / (float)(state->ticks_elapsed > 0u ? state->ticks_elapsed : 1u);
    float cargo_util_avg = 0.0f;

    if (state->cargo_util_count > 0.0f) {
        cargo_util_avg = state->cargo_util_sum / state->cargo_util_count;
    }

    out->credits = state->credits;
    out->net_profit = net_profit;
    out->profit_per_tick = profit_per_tick;
    out->survival = (destroyed || stranded) ? 0.0f : 1.0f;
    out->overheat_ticks = (float)state->overheat_ticks;
    out->pirate_encounters = (float)state->pirate_encounters;
    out->value_lost_to_pirates = state->value_lost_to_pirates;
    out->fuel_used = fmaxf(0.0f, state->fuel_start - state->fuel);
    out->hull_damage = fmaxf(0.0f, state->hull_start - state->hull);
    out->tool_wear = fmaxf(0.0f, state->tool_start - state->tool_condition);
    out->scan_count = (float)state->scan_count;
    out->mining_ticks = (float)state->mining_ticks;
    out->cargo_utilization_avg = abp_clampf(cargo_util_avg, 0.0f, 1.0f);
    out->time_remaining = state->time_remaining;
}
static void abp_pack_obs(AbpCoreState *state, float *obs_out) {
    float cargo_total = 0.0f;
    float credits_norm = 0.0f;
    float *obs = state->obs_buffer;
    uint32_t c_idx = 0;

    memset(obs, 0, sizeof(state->obs_buffer));

    cargo_total = abp_cargo_sum(state);

    obs[0] = abp_clampf(state->fuel / ABP_FUEL_MAX, 0.0f, 1.0f);
    obs[1] = abp_clampf(state->hull / ABP_HULL_MAX, 0.0f, 1.0f);
    obs[2] = abp_clampf(state->heat / ABP_HEAT_MAX, 0.0f, 1.0f);
    obs[3] = abp_clampf(state->tool_condition / ABP_TOOL_MAX, 0.0f, 1.0f);
    obs[4] = abp_clampf(cargo_total / ABP_CARGO_MAX, 0.0f, 1.0f);
    obs[5] = abp_clampf(state->alert / ABP_ALERT_MAX, 0.0f, 1.0f);
    obs[6] = abp_clampf(state->time_remaining / state->config.time_max, 0.0f, 1.0f);

    credits_norm = log1pf(fmaxf(0.0f, state->credits)) / log1pf(ABP_CREDITS_CAP);
    obs[7] = abp_clampf(credits_norm, 0.0f, 1.0f);

    for (c_idx = 0; c_idx < ABP_N_COMMODITIES; ++c_idx) {
        obs[8u + c_idx] = abp_clampf(state->cargo[c_idx] / ABP_CARGO_MAX, 0.0f, 1.0f);
    }

    obs[14] = abp_clampf((float)state->repair_kits / (float)ABP_REPAIR_KITS_CAP, 0.0f, 1.0f);
    obs[15] = abp_clampf((float)state->stabilizers / (float)ABP_STABILIZERS_CAP, 0.0f, 1.0f);
    obs[16] = abp_clampf((float)state->decoys / (float)ABP_DECOYS_CAP, 0.0f, 1.0f);

    obs[17] = abp_is_at_station(state) ? 1.0f : 0.0f;
    obs[18] = abp_selected_asteroid_valid(state) ? 1.0f : 0.0f;

    {
        uint8_t node_type = state->node_type[state->current_node];
        obs[19] = 0.0f;
        obs[20] = 0.0f;
        obs[21] = 0.0f;
        if (node_type < ABP_NODE_TYPES) {
            obs[19u + node_type] = 1.0f;
        }
    }

    obs[22] = abp_clampf((float)state->current_node * ABP_INV_MAX_NODE_INDEX, 0.0f, 1.0f);
    obs[23] = abp_clampf((float)abp_steps_to_station(state) * ABP_INV_MAX_NODE_INDEX, 0.0f, 1.0f);

    {
        uint8_t slot = 0;
        for (slot = 0; slot < ABP_MAX_NEIGHBORS; ++slot) {
            uint32_t base = 24u + 7u * slot;
            int neighbor = state->neighbors[state->current_node][slot];
            if (neighbor < 0) {
                continue;
            }

            obs[base] = 1.0f;
            {
                uint8_t neigh_type = state->node_type[neighbor];
                if (neigh_type < ABP_NODE_TYPES) {
                    obs[base + 1u + neigh_type] = 1.0f;
                }
            }

            obs[base + 4u] = abp_clampf((float)state->edge_travel_time[state->current_node][slot] *
                                            ABP_INV_TRAVEL_TIME_MAX,
                                        0.0f, 1.0f);
            obs[base + 5u] = abp_clampf(state->edge_fuel_cost[state->current_node][slot] *
                                            ABP_INV_TRAVEL_FUEL_COST_MAX,
                                        0.0f, 1.0f);
            obs[base + 6u] =
                abp_clampf(state->edge_threat_est[state->current_node][slot], 0.0f, 1.0f);
        }
    }

    {
        uint8_t a_idx = 0;
        for (a_idx = 0; a_idx < ABP_MAX_ASTEROIDS; ++a_idx) {
            uint32_t base = 68u + 11u * a_idx;

            if (state->ast_valid[state->current_node][a_idx] == 0u) {
                continue;
            }

            obs[base] = 1.0f;
            abp_normalize_comp_est_6_to_obs(state->comp_est[state->current_node][a_idx],
                                            &obs[base + 1u]);

            obs[base + 7u] =
                abp_clampf(state->stability_est[state->current_node][a_idx], 0.0f, 1.0f);
            obs[base + 8u] = abp_clampf(state->depletion[state->current_node][a_idx], 0.0f, 1.0f);
            obs[base + 9u] = abp_clampf(state->scan_conf[state->current_node][a_idx], 0.0f, 1.0f);
            obs[base + 10u] = ((int8_t)a_idx == state->selected_asteroid) ? 1.0f : 0.0f;
        }
    }

    for (c_idx = 0; c_idx < ABP_N_COMMODITIES; ++c_idx) {
        float price_norm = 0.0f;
        float d_price = 0.0f;

        if (k_price_base[c_idx] > 0.0f) {
            price_norm = state->market_price[c_idx] * k_inv_price_base[c_idx];
        }
        obs[ABP_MKT_PRICE_BASE + c_idx] = abp_clampf(price_norm, 0.0f, 1.0f);

        d_price =
            (state->market_price[c_idx] - state->market_prev_price[c_idx]) * ABP_INV_PRICE_SCALE;
        obs[ABP_MKT_DPRICE_BASE + c_idx] = abp_clampf(d_price, -1.0f, 1.0f);
    }

    obs[ABP_MKT_INV_BASE + 0u] =
        abp_clampf(state->station_inventory[0] * ABP_INV_STATION_INVENTORY_NORM_CAP, 0.0f, 1.0f);
    obs[ABP_MKT_INV_BASE + 1u] =
        abp_clampf(state->station_inventory[2] * ABP_INV_STATION_INVENTORY_NORM_CAP, 0.0f, 1.0f);
    obs[ABP_MKT_INV_BASE + 2u] =
        abp_clampf(state->station_inventory[3] * ABP_INV_STATION_INVENTORY_NORM_CAP, 0.0f, 1.0f);
    obs[ABP_MKT_INV_BASE + 3u] =
        abp_clampf(state->station_inventory[4] * ABP_INV_STATION_INVENTORY_NORM_CAP, 0.0f, 1.0f);

    if (obs_out != NULL) {
        memcpy(obs_out, obs, sizeof(state->obs_buffer));
    }
}

void abp_core_default_config(AbpCoreConfig *config) {
    if (config == NULL) {
        return;
    }

    config->time_max = ABP_TIME_MAX;
    config->invalid_action_penalty = 0.01f;
}

AbpCoreState *abp_core_create(const AbpCoreConfig *config, uint64_t seed) {
    AbpCoreState *state = (AbpCoreState *)malloc(sizeof(AbpCoreState));
    if (state == NULL) {
        return NULL;
    }

    abp_core_init(state, config, seed);
    return state;
}

void abp_core_destroy(AbpCoreState *state) {
    if (state == NULL) {
        return;
    }

    free(state);
}

void abp_core_init(AbpCoreState *state, const AbpCoreConfig *config, uint64_t seed) {
    uint32_t c_idx = 0;

    if (state == NULL) {
        return;
    }

    memset(state, 0, sizeof(*state));

    if (config != NULL) {
        state->config = *config;
    } else {
        abp_core_default_config(&state->config);
    }

    if (state->config.time_max <= 0.0f) {
        state->config.time_max = ABP_TIME_MAX;
    }
    if (state->config.invalid_action_penalty <= 0.0f) {
        state->config.invalid_action_penalty = 0.01f;
    }

    state->seed = seed;
    state->selected_asteroid = -1;

    state->fuel = ABP_FUEL_MAX;
    state->hull = ABP_HULL_MAX;
    state->heat = 0.0f;
    state->tool_condition = ABP_TOOL_MAX;
    state->alert = 0.0f;
    state->time_remaining = state->config.time_max;
    state->credits = 0.0f;

    state->repair_kits = 3u;
    state->stabilizers = 2u;
    state->decoys = 1u;

    state->fuel_start = state->fuel;
    state->hull_start = state->hull;
    state->tool_start = state->tool_condition;

    for (c_idx = 0; c_idx < ABP_N_COMMODITIES; ++c_idx) {
        state->cargo[c_idx] = 0.0f;
    }

    abp_rng_seed(&state->rng, seed, 54u);
    abp_generate_world(state);
    abp_pack_obs(state, NULL);
}

void abp_core_reset(AbpCoreState *state, uint64_t seed, float *obs_out) {
    AbpCoreConfig cfg;

    if (state == NULL) {
        return;
    }

    cfg = state->config;
    abp_core_init(state, &cfg, seed);
    abp_pack_obs(state, obs_out);
}

void abp_core_step(AbpCoreState *state, uint8_t action, AbpCoreStepResult *out) {
    uint8_t terminated = 0u;
    uint8_t truncated = 0u;
    uint8_t invalid_action = 0u;
    uint16_t dt = 1u;
    uint8_t action_int = action;
    float reward = 0.0f;
    float cargo_value_after = 0.0f;
    uint8_t destroyed = 0u;
    uint8_t stranded = 0u;
    uint8_t done = 0u;
    AbpStepSnapshot snapshot;

    if (state == NULL || out == NULL) {
        return;
    }

    memset(out, 0, sizeof(*out));

    if (state->needs_reset > 0u) {
        abp_pack_obs(state, out->obs);
        out->action = -1;
        out->terminated = 1u;
        out->truncated = 0u;
        out->invalid_action = 1u;
        out->dt = 0u;
        abp_fill_step_metrics(state, out, 0u, 0u);
        return;
    }

    snapshot.credits_before = state->credits;
    snapshot.fuel_before = state->fuel;
    snapshot.hull_before = state->hull;
    snapshot.heat_before = state->heat;
    snapshot.tool_before = state->tool_condition;
    snapshot.cargo_value_before = abp_est_cargo_value(state);
    snapshot.value_lost_to_pirates_before = state->value_lost_to_pirates;

    if (action_int >= ABP_N_ACTIONS) {
        invalid_action = 1u;
        action_int = 6u;
    }

    if (action_int <= 5u) {
        AbpTravelResult travel = abp_apply_travel(state, action_int);
        dt = travel.dt;
        if (travel.invalid > 0u) {
            invalid_action = 1u;
        }
    } else if (action_int == 6u) {
        abp_apply_hold(state);
    } else if (action_int == 7u) {
        abp_apply_emergency_burn(state);
    } else if (action_int == 8u) {
        dt = ABP_WIDE_SCAN_TIME;
        state->fuel -= ABP_WIDE_SCAN_FUEL;
        state->alert += ABP_WIDE_SCAN_ALERT;
        abp_update_cluster_priors_with_noise(state);
        state->scan_count += 1u;
    } else if (action_int == 9u) {
        dt = ABP_FOCUSED_SCAN_TIME;
        state->fuel -= ABP_FOCUSED_SCAN_FUEL;
        state->alert += ABP_FOCUSED_SCAN_ALERT;
        if (!abp_selected_asteroid_valid(state)) {
            invalid_action = 1u;
        } else {
            abp_update_asteroid_estimates(state, (uint8_t)state->selected_asteroid, 1u);
            state->scan_count += 1u;
        }
    } else if (action_int == 10u) {
        dt = ABP_DEEP_SCAN_TIME;
        state->fuel -= ABP_DEEP_SCAN_FUEL;
        state->alert += ABP_DEEP_SCAN_ALERT;
        if (!abp_selected_asteroid_valid(state)) {
            invalid_action = 1u;
        } else {
            abp_update_asteroid_estimates(state, (uint8_t)state->selected_asteroid, 2u);
            state->scan_count += 1u;
        }
    } else if (action_int == 11u) {
        dt = ABP_THREAT_LISTEN_TIME;
        abp_update_neighbor_threat_estimates(state);
    } else if (action_int >= 12u && action_int <= 27u) {
        if (!abp_select_asteroid(state, (int)action_int - 12)) {
            invalid_action = 1u;
        }
    } else if (action_int >= 28u && action_int <= 30u) {
        if (!abp_selected_asteroid_valid(state)) {
            invalid_action = 1u;
        } else {
            abp_mine_selected(state, action_int);
        }
    } else if (action_int == 31u) {
        dt = ABP_STABILIZE_TIME;
        if (!abp_selected_asteroid_valid(state) || state->stabilizers == 0u) {
            invalid_action = 1u;
        } else {
            state->stabilizers -= 1u;
            state->stabilize_buff_ticks[(uint8_t)state->selected_asteroid] =
                ABP_STABILIZE_BUFF_TICKS;
        }
    } else if (action_int == 32u) {
        dt = ABP_REFINE_TIME;
        state->fuel -= ABP_REFINE_FUEL;
        state->heat += ABP_REFINE_HEAT;
        state->alert += ABP_REFINE_ALERT;
        abp_refine_some_cargo(state);
    } else if (action_int == 33u) {
        dt = ABP_COOLDOWN_TIME;
        state->fuel -= ABP_COOLDOWN_FUEL;
        state->heat = fmaxf(0.0f, state->heat - ABP_COOLDOWN_AMOUNT);
        state->alert += ABP_COOLDOWN_ALERT;
    } else if (action_int == 34u) {
        dt = ABP_MAINT_TIME;
        if (state->repair_kits == 0u) {
            invalid_action = 1u;
        } else {
            state->repair_kits -= 1u;
            state->tool_condition =
                fminf(ABP_TOOL_MAX, state->tool_condition + ABP_TOOL_REPAIR_AMOUNT);
        }
    } else if (action_int == 35u) {
        dt = ABP_PATCH_TIME;
        if (state->repair_kits == 0u) {
            invalid_action = 1u;
        } else {
            state->repair_kits -= 1u;
            state->hull = fminf(ABP_HULL_MAX, state->hull + ABP_HULL_PATCH_AMOUNT);
        }
    } else if (action_int >= 36u && action_int <= 41u) {
        uint8_t c_idx = action_int - 36u;
        state->cargo[c_idx] = 0.0f;
        state->alert = fmaxf(0.0f, state->alert - ABP_JETTISON_ALERT_RELIEF);
    } else if (action_int == 42u) {
        dt = ABP_DOCK_TIME;
        if (!abp_is_at_station(state)) {
            invalid_action = 1u;
        } else {
            state->alert = fmaxf(0.0f, state->alert - ABP_DOCK_ALERT_DROP);
        }
    } else if (action_int >= 43u && action_int <= 60u) {
        if (!abp_is_at_station(state)) {
            invalid_action = 1u;
        } else {
            abp_sell_action(state, action_int);
        }
    } else if (action_int >= 61u && action_int <= 66u) {
        if (!abp_is_at_station(state) || !abp_purchase_station_item(state, action_int)) {
            invalid_action = 1u;
        }
    } else if (action_int == 67u) {
        dt = ABP_OVERHAUL_TIME;
        if (!abp_is_at_station(state) || state->credits < ABP_OVERHAUL_COST) {
            invalid_action = 1u;
        } else {
            state->credits -= ABP_OVERHAUL_COST;
            state->total_spend += ABP_OVERHAUL_COST;
            state->hull = ABP_HULL_MAX;
            state->tool_condition = ABP_TOOL_MAX;
        }
    } else if (action_int == 68u) {
        terminated = 1u;
    } else {
        invalid_action = 1u;
    }
    if (invalid_action > 0u) {
        dt = 1u;
        abp_apply_hold(state);
    }

    abp_apply_global_dynamics(state, dt);
    state->ticks_elapsed += dt;

    destroyed = state->hull <= 0.0f ? 1u : 0u;
    stranded = (state->fuel <= 0.0f && !abp_is_at_station(state)) ? 1u : 0u;

    if (destroyed || stranded) {
        terminated = 1u;
    }

    if (state->time_remaining <= 0.0f && terminated == 0u) {
        truncated = 1u;
    }

    done = (terminated || truncated) ? 1u : 0u;

    cargo_value_after = abp_est_cargo_value(state);
    reward = abp_compute_reward(state, &snapshot, cargo_value_after, action_int, dt, invalid_action,
                                destroyed, stranded, done);

    abp_pack_obs(state, out->obs);

    out->reward = reward;
    out->terminated = terminated;
    out->truncated = truncated;
    out->invalid_action = invalid_action;
    out->dt = dt;
    out->action = (int16_t)action_int;

    abp_fill_step_metrics(state, out, destroyed, stranded);

    state->needs_reset = done;
}

void abp_core_reset_many(AbpCoreState **states, const uint64_t *seeds, uint32_t count,
                         float *obs_out) {
    uint32_t i = 0;

    if (states == NULL || count == 0u) {
        return;
    }

    for (i = 0; i < count; ++i) {
        AbpCoreState *state = states[i];
        uint64_t seed = 0u;
        float *obs_ptr = NULL;

        if (state == NULL) {
            continue;
        }

        seed = (seeds != NULL) ? seeds[i] : state->seed;
        if (obs_out != NULL) {
            obs_ptr = obs_out + ((size_t)i * (size_t)ABP_OBS_DIM);
        }

        abp_core_reset(state, seed, obs_ptr);
    }
}

void abp_core_step_many(AbpCoreState **states, const uint8_t *actions, uint32_t count,
                        AbpCoreStepResult *out_results) {
    uint32_t i = 0;

    if (states == NULL || count == 0u) {
        return;
    }

    for (i = 0; i < count; ++i) {
        AbpCoreState *state = states[i];
        uint8_t action = 6u;

        if (state == NULL) {
            continue;
        }

        if (actions != NULL) {
            action = actions[i];
        }

        if (out_results != NULL) {
            abp_core_step(state, action, &out_results[i]);
        } else {
            AbpCoreStepResult scratch;
            abp_core_step(state, action, &scratch);
        }
    }
}
