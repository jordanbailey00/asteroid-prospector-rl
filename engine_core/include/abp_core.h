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

#define ABP_CREDIT_SCALE 1000.0f
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
    int16_t action;

    float credits;
    float net_profit;
    float profit_per_tick;
    float survival;
    float overheat_ticks;
    float pirate_encounters;
    float value_lost_to_pirates;
    float fuel_used;
    float hull_damage;
    float tool_wear;
    float scan_count;
    float mining_ticks;
    float cargo_utilization_avg;
    float time_remaining;
} AbpCoreStepResult;

typedef struct AbpCoreState {
    AbpCoreConfig config;
    AbpRng rng;
    uint64_t seed;

    uint32_t ticks_elapsed;
    float time_remaining;
    uint8_t needs_reset;

    uint8_t node_count;
    uint8_t current_node;
    int8_t selected_asteroid;

    float credits;
    float fuel;
    float hull;
    float heat;
    float tool_condition;
    float alert;
    float cargo[ABP_N_COMMODITIES];

    uint8_t repair_kits;
    uint8_t stabilizers;
    uint8_t decoys;
    uint8_t escape_buff_ticks;
    uint8_t stabilize_buff_ticks[ABP_MAX_ASTEROIDS];

    uint8_t node_type[ABP_MAX_NODES];
    float node_hazard[ABP_MAX_NODES];
    float node_pirate[ABP_MAX_NODES];
    int8_t neighbors[ABP_MAX_NODES][ABP_MAX_NEIGHBORS];
    uint8_t edge_travel_time[ABP_MAX_NODES][ABP_MAX_NEIGHBORS];
    float edge_fuel_cost[ABP_MAX_NODES][ABP_MAX_NEIGHBORS];
    float edge_threat_true[ABP_MAX_NODES][ABP_MAX_NEIGHBORS];
    float edge_threat_est[ABP_MAX_NODES][ABP_MAX_NEIGHBORS];

    uint8_t ast_valid[ABP_MAX_NODES][ABP_MAX_ASTEROIDS];
    float true_comp[ABP_MAX_NODES][ABP_MAX_ASTEROIDS][ABP_N_COMMODITIES];
    float richness[ABP_MAX_NODES][ABP_MAX_ASTEROIDS];
    float stability_true[ABP_MAX_NODES][ABP_MAX_ASTEROIDS];
    float noise_profile[ABP_MAX_NODES][ABP_MAX_ASTEROIDS];

    float comp_est[ABP_MAX_NODES][ABP_MAX_ASTEROIDS][ABP_N_COMMODITIES];
    float stability_est[ABP_MAX_NODES][ABP_MAX_ASTEROIDS];
    float scan_conf[ABP_MAX_NODES][ABP_MAX_ASTEROIDS];
    float depletion[ABP_MAX_NODES][ABP_MAX_ASTEROIDS];

    float market_price[ABP_N_COMMODITIES];
    float market_prev_price[ABP_N_COMMODITIES];
    float price_phase[ABP_N_COMMODITIES];
    float price_period[ABP_N_COMMODITIES];
    float price_amp[ABP_N_COMMODITIES];
    float station_inventory[ABP_N_COMMODITIES];
    float recent_sales[ABP_N_COMMODITIES];

    float total_spend;
    uint32_t overheat_ticks;
    uint32_t pirate_encounters;
    float value_lost_to_pirates;
    uint32_t scan_count;
    uint32_t mining_ticks;
    float fuel_start;
    float hull_start;
    float tool_start;
    float cargo_util_sum;
    float cargo_util_count;

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
