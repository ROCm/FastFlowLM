/// \file hrx_coh_ranges.hpp
/// \brief Dirty-range set for one HRX allocation shared by sub-buffer slices.
#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

namespace hrx {

struct coh_range {
    size_t offset = 0;
    size_t length = 0;
};

inline size_t coh_range_end(const coh_range& range) {
    return range.offset + range.length;
}

/// Record [offset, offset + length) as dirty. Overlapping and adjacent ranges merge.
inline void coh_range_add(std::vector<coh_range>& ranges, size_t offset, size_t length) {
    if (length == 0) return;
    size_t begin = offset;
    size_t end = offset + length;
    std::vector<coh_range> kept;
    kept.reserve(ranges.size() + 1);
    for (const coh_range& range : ranges) {
        size_t range_begin = range.offset;
        size_t range_end = coh_range_end(range);
        if (range_end < begin || range_begin > end) {
            kept.push_back(range);
            continue;
        }
        begin = std::min(begin, range_begin);
        end = std::max(end, range_end);
    }
    kept.push_back(coh_range{begin, end - begin});
    ranges.swap(kept);
}

/// Remove the intersection of `ranges` with [offset, offset + length).
/// The returned ranges are that intersection, in the order they were stored.
inline std::vector<coh_range> coh_range_take(
    std::vector<coh_range>& ranges, size_t offset, size_t length) {
    std::vector<coh_range> taken;
    if (length == 0) return taken;
    size_t begin = offset;
    size_t end = offset + length;
    std::vector<coh_range> kept;
    kept.reserve(ranges.size());
    for (const coh_range& range : ranges) {
        size_t range_begin = range.offset;
        size_t range_end = coh_range_end(range);
        size_t overlap_begin = std::max(begin, range_begin);
        size_t overlap_end = std::min(end, range_end);
        if (overlap_begin >= overlap_end) {
            kept.push_back(range);
            continue;
        }
        taken.push_back(coh_range{overlap_begin, overlap_end - overlap_begin});
        if (range_begin < overlap_begin) {
            kept.push_back(coh_range{range_begin, overlap_begin - range_begin});
        }
        if (overlap_end < range_end) {
            kept.push_back(coh_range{overlap_end, range_end - overlap_end});
        }
    }
    ranges.swap(kept);
    return taken;
}

}  // namespace hrx
