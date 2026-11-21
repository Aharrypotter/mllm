#ifndef HMX_I8_PACKED_TOPK_LAYOUT_H
#define HMX_I8_PACKED_TOPK_LAYOUT_H

#include <stddef.h>
#include <stdint.h>

/* The CPU-packed shared-key RPC reads K from its packed arena. Each
 * unused raw-K slab can hold compact indices followed by the row offsets.
 * Keep the offsets on their own cache lines, away from the DSP output. */
static inline size_t hmx_i8_topk_offsets_start(size_t slab_bytes, int rows) {
    return slab_bytes - (((size_t)(rows + 1) * sizeof(int32_t) + 127) & ~(size_t)127);
}

#endif
