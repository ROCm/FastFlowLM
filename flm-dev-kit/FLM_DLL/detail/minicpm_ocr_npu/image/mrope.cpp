// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file image/mrope.cpp
#include "image/mrope.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace mco {

void llm_grid(int n_tokens, int vh, int vw, int& h, int& w) {
    h = w = 0;
    if (n_tokens <= 0 || vh <= 0 || vw <= 0) return;
    const int vit_total = vh * vw;
    if (vit_total == n_tokens) { h = vh; w = vw; return; }
    if (vit_total < n_tokens) return;
    // Python's round() is half-to-even; nearbyint, not lround.
    const int factor = (int)std::nearbyint(std::sqrt((double)vit_total / n_tokens));
    if (factor > 0 && vh % factor == 0 && vw % factor == 0) {
        const int a = vh / factor, b = vw / factor;
        if (a * b == n_tokens) { h = a; w = b; return; }
    }
    const double ratio = (double)vh / vw;
    const int ww = std::max(1, (int)std::nearbyint(std::sqrt(n_tokens / ratio)));
    const int hh = n_tokens / ww;
    if (hh * ww == n_tokens && hh > 0) { h = hh; w = ww; }
}

namespace {

/// `torch.linspace(0, end, n).round()` for float32.
///
/// ATen's CPU linspace fills from both ends -- `start + step*i` below the
/// halfway index, `end - step*(n-1-i)` from it -- with a float step. Checked
/// against the installed torch for every end < 200 and n < 80: the rounded
/// integers agree in all 15,800 cases, ties included. (The float VALUES differ
/// in the last bit in some cases -- the vectorised arm -- but never across a
/// rounding boundary in that range.) Products go through volatiles so
/// -ffp-contract cannot fuse them.
std::vector<int> linspace_round(int end, int n) {
    std::vector<int> out((size_t)n, 0);
    if (n == 1) return out;
    const float fend = (float)end;
    const float step = fend / (float)(n - 1);
    const int half = n / 2;
    for (int i = 0; i < n; i++) {
        float v;
        if (i < half) { volatile float p = step * (float)i;           v = 0.0f + p; }
        else          { volatile float p = step * (float)(n - 1 - i); v = fend - p; }
        out[(size_t)i] = (int)std::nearbyint(v);   // half-to-even, as torch.round
    }
    return out;
}

struct segment { int bs, be, gi; };
struct group   { segment thumb; std::vector<segment> slices; };

}  // namespace

int canvas_mrope(const int* tokens, int L, const canvas_token_ids& ids,
                 const std::vector<view_grid>& views, int pos0, int* out) {
    auto set3 = [&](int p, int t, int h, int w) {
        out[3 * p + 0] = t; out[3 * p + 1] = h; out[3 * p + 2] = w;
    };
    auto structural = [&](int tok) {
        return tok == ids.im_start || tok == ids.im_end ||
               tok == ids.slice_start || tok == ids.slice_end;
    };

    // ---- build_image_bounds(): (first token, one-past-last) of each segment
    std::vector<int> starts, ends;
    for (int p = 0; p < L; p++) {
        if (tokens[p] == ids.im_start || tokens[p] == ids.slice_start) starts.push_back(p + 1);
        if (tokens[p] == ids.im_end   || tokens[p] == ids.slice_end)   ends.push_back(p);
    }
    const size_t nseg = std::min(starts.size(), ends.size());
    if (nseg != views.size())
        throw std::runtime_error("canvas_mrope: the prompt has " + std::to_string(nseg) +
                                 " image segments but " + std::to_string(views.size()) +
                                 " views were supplied");

    if (nseg == 0) {
        for (int p = 0; p < L; p++) set3(p, pos0 + p, pos0 + p, pos0 + p);
        return pos0 + L;
    }

    // ---- group: a thumbnail and the slices that follow it
    std::vector<group> groups;
    for (size_t i = 0; i < nseg; i++) {
        const segment s{starts[i], ends[i], (int)i};
        const int marker = s.bs - 1;
        const bool is_slice = marker >= 0 && tokens[marker] == ids.slice_start;
        if (is_slice && !groups.empty()) groups.back().slices.push_back(s);
        else                             groups.push_back(group{s, {}});
    }

    auto group_end = [&](const group& g) {
        int ge = g.slices.empty() ? g.thumb.be : g.slices.back().be;
        while (ge < L && structural(tokens[ge])) ge++;
        return std::min(ge, L);
    };

    // HF merges groups whose gap is only structural tokens and "\n" into a
    // video group with a different layout. Unreachable with image ids in the
    // prompt; refuse it rather than lay it out the image way.
    for (size_t g = 1; g < groups.size(); g++) {
        const int ge = group_end(groups[g - 1]);
        const int ns = groups[g].thumb.bs - 1;
        bool tight = true;
        for (int p = ge; p < ns && tight; p++)
            tight = structural(tokens[p]) || tokens[p] == ids.newline;
        if (tight)
            throw std::runtime_error(
                "canvas_mrope: two image groups with only structural tokens between them -- "
                "HF lays that out as a VIDEO group, which this engine does not implement. "
                "Each image must be preceded by <image_id>N</image_id>.");
    }

    // ---- assign positions
    int pos = pos0, cursor = 0;
    for (const group& g : groups) {
        const segment& th = g.thumb;
        const int gs = th.bs - 1;
        const int ge = group_end(g);

        for (int p = cursor; p < gs; p++) set3(p, pos + (p - cursor), pos + (p - cursor), pos + (p - cursor));
        pos += std::max(0, gs - cursor);

        const int base = pos;
        const int halo_lo = std::max(base - 1, 0);

        const int n_thumb = th.be - th.bs;
        int lth, ltw;
        llm_grid(n_thumb, views[(size_t)th.gi].h, views[(size_t)th.gi].w, lth, ltw);

        int rows = 0, cols = 0, sh = 0, sw = 0, H, W;
        if (!g.slices.empty()) {
            const int ns = (int)g.slices.size();
            cols = ns;
            for (int k = 0; k + 1 < ns; k++)
                if (g.slices[(size_t)k + 1].bs - g.slices[(size_t)k].be > 2) { cols = k + 1; break; }
            rows = cols > 0 ? ns / cols : 1;
            if (rows * cols != ns) { rows = 1; cols = ns; }
            const segment& s0 = g.slices[0];
            llm_grid(s0.be - s0.bs, views[(size_t)s0.gi].h, views[(size_t)s0.gi].w, sh, sw);
            H = rows * sh;
            W = cols * sw;
        } else {
            H = std::max(lth, 1);
            W = std::max(ltw, 1);
        }

        // base coat, then the halos
        for (int p = gs; p < ge; p++) set3(p, base, base, base);
        out[3 * gs + 1] = halo_lo;
        out[3 * gs + 2] = halo_lo;
        if (th.be < ge) { out[3 * th.be + 1] = base + H; out[3 * th.be + 2] = base + W; }

        // thumbnail tokens: stretched over the canvas when there are slices
        if (lth > 0 && ltw > 0 && lth * ltw == n_thumb) {
            std::vector<int> hc, wc;
            if (!g.slices.empty() && H > 0 && W > 0) {
                hc = linspace_round(H - 1, lth);
                wc = linspace_round(W - 1, ltw);
            } else {
                hc.resize((size_t)lth); wc.resize((size_t)ltw);
                for (int i = 0; i < lth; i++) hc[(size_t)i] = i;
                for (int i = 0; i < ltw; i++) wc[(size_t)i] = i;
            }
            for (int r = 0; r < lth; r++)
                for (int c = 0; c < ltw; c++)
                    set3(th.bs + r * ltw + c, base, base + hc[(size_t)r], base + wc[(size_t)c]);
        } else {
            for (int i = 0; i < n_thumb; i++) set3(th.bs + i, base + i, base + i, base + i);
        }

        // slice tokens and their brackets
        for (size_t k = 0; k < g.slices.size(); k++) {
            const segment& s = g.slices[k];
            const int n_s = s.be - s.bs;
            int shk, swk;
            llm_grid(n_s, views[(size_t)s.gi].h, views[(size_t)s.gi].w, shk, swk);
            const int h_off = ((int)k / cols) * sh;
            const int w_off = ((int)k % cols) * sw;

            const int ss = s.bs - 1;
            if (ss >= gs) { out[3 * ss + 1] = base + h_off; out[3 * ss + 2] = base + w_off; }
            const int end_h = (shk > 0 && swk > 0) ? h_off + shk - 1 : h_off;
            const int end_w = (shk > 0 && swk > 0) ? w_off + swk - 1 : w_off;
            if (s.be < ge) { out[3 * s.be + 1] = base + end_h; out[3 * s.be + 2] = base + end_w; }

            if (shk > 0 && swk > 0 && shk * swk == n_s) {
                for (int r = 0; r < shk; r++)
                    for (int c = 0; c < swk; c++)
                        set3(s.bs + r * swk + c, base, base + h_off + r, base + w_off + c);
            } else {
                for (int i = 0; i < n_s; i++) set3(s.bs + i, base + i, base + i, base + i);
            }
        }

        // "\n" between slice rows: right of the canvas, on the row that ended
        for (size_t k = 0; k + 1 < g.slices.size(); k++) {
            const int gap_s = g.slices[k].be;
            const int gap_e = g.slices[k + 1].bs;
            if (gap_e - gap_s <= 2) continue;
            const int boundary_h = ((int)k / cols + 1) * sh - 1;
            for (int p = gap_s + 1; p < gap_e - 1; p++) {
                out[3 * p + 1] = base + boundary_h;
                out[3 * p + 2] = base + cols * sw;
            }
        }

        pos = base + std::max(H, W) + 1;
        cursor = ge;
    }

    for (int p = cursor; p < L; p++) set3(p, pos + (p - cursor), pos + (p - cursor), pos + (p - cursor));
    return pos + std::max(0, L - cursor);
}

}  // namespace mco
