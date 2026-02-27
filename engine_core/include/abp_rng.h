#ifndef ABP_RNG_H
#define ABP_RNG_H

#include <stdint.h>

typedef struct AbpRng {
    uint64_t state;
    uint64_t inc;
} AbpRng;

void abp_rng_seed(AbpRng *rng, uint64_t seed, uint64_t stream);
uint32_t abp_rng_next_u32(AbpRng *rng);
float abp_rng_next_f32(AbpRng *rng);

#endif
