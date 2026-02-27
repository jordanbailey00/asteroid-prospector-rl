#include "abp_rng.h"

static uint32_t abp_pcg32_next(AbpRng *rng) {
    uint64_t oldstate = rng->state;
    uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = (uint32_t)(oldstate >> 59u);

    rng->state = oldstate * 6364136223846793005ULL + rng->inc;
    return (xorshifted >> rot) | (xorshifted << ((-(int32_t)rot) & 31));
}

void abp_rng_seed(AbpRng *rng, uint64_t seed, uint64_t stream) {
    rng->state = 0U;
    rng->inc = (stream << 1u) | 1u;
    (void)abp_pcg32_next(rng);
    rng->state += seed;
    (void)abp_pcg32_next(rng);
}

uint32_t abp_rng_next_u32(AbpRng *rng) { return abp_pcg32_next(rng); }

float abp_rng_next_f32(AbpRng *rng) {
    /* 2^-32 precision in [0,1). */
    return (float)(abp_rng_next_u32(rng) / 4294967296.0);
}
