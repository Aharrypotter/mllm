#include "dsp/hvx_score_topk.h"

#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <string.h>

enum { group_rows = 4, tile_cols = 32, bins = 256, bins_per_vector = 8, vector_bytes = 128 };
typedef uint16_t LaneHistograms[bins / bins_per_vector][group_rows][2][bins_per_vector];

extern void hmx_i8_hvx_histogram_tiles(const uint8_t *tiles, unsigned count, LaneHistograms lanes);

static unsigned tile_offset(int lane, int col) {
    return (unsigned)((col / tile_cols) * vector_bytes + lane * tile_cols + col % tile_cols);
}

typedef uint16_t GroupTotals[bins / bins_per_vector][group_rows];
typedef union {
    LaneHistograms banks;
    HVX_Vector vectors[bins / bins_per_vector];
} HistogramStorage;

static void tile_histogram(const uint8_t *scores, int visible, HistogramStorage *hist, GroupTotals totals) {
    const int complete = visible / tile_cols;
    if (complete > 0) {
        hmx_i8_hvx_histogram_tiles(scores, (unsigned)complete, hist->banks);
        for (int group = 0; group < bins / bins_per_vector; ++group) {
            const HVX_Vector banks = hist->vectors[group];
            // Each row occupies 32 bytes: two consecutive eight-bin banks.
            // Fold into the first bank, then sum that bank's eight bins.
            const HVX_Vector merged = Q6_Vh_vadd_VhVh(banks, Q6_V_vror_VR(banks, 16));
            hist->vectors[group] = merged;
            const HVX_Vector pairs = Q6_Vh_vadd_VhVh(merged, Q6_V_vror_VR(merged, 2));
            const HVX_Vector quads = Q6_Vh_vadd_VhVh(pairs, Q6_V_vror_VR(pairs, 4));
            const HVX_Vector sums = Q6_Vh_vadd_VhVh(quads, Q6_V_vror_VR(quads, 8));
            for (int lane = 0; lane < group_rows; ++lane) {
                totals[group][lane] = (uint16_t)Q6_R_vextract_VR(sums, lane * 32);
            }
        }
    } else {
        memset(hist, 0, sizeof(*hist));
        memset(totals, 0, sizeof(GroupTotals));
    }
    for (int lane = 0; lane < group_rows; ++lane) {
        for (int col = complete * tile_cols; col < visible + lane; ++col) {
            const unsigned score = scores[tile_offset(lane, col)];
            const unsigned group = score / bins_per_vector;
            ++hist->banks[group][lane][0][score % bins_per_vector];
            ++totals[group][lane];
        }
    }
}

static int select_cutoff(const LaneHistograms hist, const GroupTotals totals, int lane, int *remaining) {
    enum { groups_per_sign = bins / (2 * bins_per_vector) };
    for (int group = groups_per_sign - 1; group >= -groups_per_sign; --group) {
        const unsigned hardware_group = (unsigned)group & (bins / bins_per_vector - 1);
        const int count = totals[hardware_group][lane];
        if (*remaining > count) {
            *remaining -= count;
            continue;
        }
        const uint16_t *bank = hist[hardware_group][lane][0];
        for (int bin = bins_per_vector - 1; bin >= 0; --bin) {
            if (*remaining <= bank[bin]) return group * bins_per_vector + bin;
            *remaining -= bank[bin];
        }
        break;
    }
    return -129;
}

static inline HVX_Vector pack_predicate(HVX_VectorPred predicate) {
    // Match the four-row HMX tile. A 4-byte dot product makes a nibble;
    // a 2-halfword dot product combines two adjacent nibbles into a byte.
    const HVX_Vector weighted = Q6_V_vand_QR(predicate, 0x08040201);
    const HVX_Vector words = Q6_Vuw_vrmpy_VubRub(weighted, 0x01010101);
    const HVX_Vector nibbles = Q6_Vh_vpacke_VwVw(words, words);
    const HVX_Vector masks = Q6_Vw_vdmpy_VhRh_sat(nibbles, 0x00100001);
    const HVX_Vector halfwords = Q6_Vh_vpacke_VwVw(masks, masks);
    return Q6_Vb_vpacke_VhVh(halfwords, halfwords);
}

static inline uint32_t row_mask(HVX_Vector packed, int lane) {
    return (uint32_t)Q6_R_vextract_VR(packed, lane * 4);
}

static uint32_t take_equal_prefix(uint32_t equal, int *remaining) {
    const int count = __builtin_popcount(equal);
    if (count <= *remaining) {
        *remaining -= count;
        return equal;
    }
    uint32_t result = 0;
    while (*remaining > 0) {
        const uint32_t lowest = equal & (0u - equal);
        result |= lowest;
        equal ^= lowest;
        --*remaining;
    }
    return result;
}

static int emit_four_rows(const uint8_t *block, int first_visible, const int thresholds[group_rows],
                          int ties[group_rows], const int keep[group_rows], uint16_t *out[group_rows]) {
    union { HVX_Vector vector; uint8_t bytes[vector_bytes]; } cutoffs;
    for (int lane = 0; lane < group_rows; ++lane)
        memset(cutoffs.bytes + lane * tile_cols, (uint8_t)thresholds[lane], tile_cols);
    int emitted[group_rows] = {0};
    // The final tile may be only partially visible or absent in the first rows.
    for (int column = 0; column < first_visible + group_rows - 1; column += tile_cols) {
        HVX_Vector scores;
        memcpy(&scores, block + (column / tile_cols) * vector_bytes, sizeof(scores));
        const HVX_Vector greater = pack_predicate(Q6_Q_vcmp_gt_VbVb(scores, cutoffs.vector));
        const HVX_Vector equal = pack_predicate(Q6_Q_vcmp_eq_VbVb(scores, cutoffs.vector));
        for (int lane = 0; lane < group_rows; ++lane) {
            const int left = first_visible + lane - column;
            if (left <= 0) continue;
            const uint32_t valid = left >= tile_cols ? UINT32_MAX : UINT32_MAX >> (tile_cols - left);
            uint32_t selected = row_mask(greater, lane) & valid;
            if (ties[lane] > 0) {
                const uint32_t matching = row_mask(equal, lane) & valid;
                selected |= take_equal_prefix(matching, &ties[lane]);
            }
            if (__builtin_popcount(selected) > keep[lane] - emitted[lane]) return -1;
            while (selected != 0) {
                const int bit = __builtin_ctz(selected);
                out[lane][emitted[lane]++] = (uint16_t)(column + bit);
                selected &= selected - 1;
            }
        }
    }
    for (int lane = 0; lane < group_rows; ++lane)
        if (emitted[lane] != keep[lane] || ties[lane] != 0) return -1;
    return 0;
}

int hmx_i8_hvx_topk_native(const uint8_t *scores, int rows, int padded_columns,
                           int valid_columns, const int32_t *offsets,
                           uint16_t *indices, size_t capacity) {
    if (scores == NULL || offsets == NULL || indices == NULL ||
        ((uintptr_t)scores & (vector_bytes - 1)) != 0 ||
        rows <= 0 || rows % group_rows != 0 ||
        valid_columns < rows || valid_columns > UINT16_MAX ||
        padded_columns < valid_columns || padded_columns % tile_cols != 0 ||
        offsets[0] != 0 || offsets[rows] <= 0 ||
        (size_t)offsets[rows] > capacity) {
        return -1;
    }
    for (int row = 0; row < rows; ++row) {
        const int previous = offsets[row];
        const int next = offsets[row + 1];
        if (next <= previous || next > offsets[rows] ||
            next - previous > valid_columns - rows + row + 1) {
            return -1;
        }
    }

    for (int group = 0; group < rows; group += group_rows) {
        const int visible = valid_columns - rows + group + 1;
        const uint8_t *block = scores + (size_t)group * padded_columns;
        _Alignas(vector_bytes) HistogramStorage histogram;
        GroupTotals totals;
        int thresholds[group_rows], ties[group_rows], keep[group_rows];
        uint16_t *out[group_rows];
        tile_histogram(block, visible, &histogram, totals);
        for (int lane = 0; lane < group_rows; ++lane) {
            const int row = group + lane;
            keep[lane] = offsets[row + 1] - offsets[row];
            out[lane] = indices + offsets[row];
            int remaining = keep[lane];
            const int threshold = select_cutoff(histogram.banks, totals, lane, &remaining);
            if (threshold < -128) return -1;
            thresholds[lane] = threshold;
            ties[lane] = remaining;
        }
        if (emit_four_rows(block, visible, thresholds, ties, keep, out) != 0) {
            return -1;
        }
    }
    return 0;
}
