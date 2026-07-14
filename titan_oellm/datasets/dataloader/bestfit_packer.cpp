/* Best-fit packing plan builder for BestFitPackedDataset.
 *
 * Builds a pre-computed packing plan that maps global sequences to their
 * constituent document fragments.
 *
 * Algorithm:
 *   1. FILL a BST buffer (tree) to tree_size fragments from the document stream.
 *      - Docs >= 2*tokens_per_seq: extract full-sequence chunks, emit directly.
 *      - Docs between tokens_per_seq and 2*tokens_per_seq: split in half → tree.
 *      - Docs < tokens_per_seq (and >= min_seq_len): insert into tree.
 *   2. EMIT one sequence using oldest-first + subset-sum DP:
 *      a. Pop the oldest fragment (FIFO) as the first pick (prevents starvation).
 *      b. Subset-sum DP (bitset) on remaining tree fragments to find the optimal
 *         subset summing as close to (tokens_per_seq - first_frag_len) as possible.
 *      c. Any remaining gap is left as padding (handled by collator).
 *   3. Refill tree when it drops below tree_size; repeat until all docs consumed.
 *
 * Data structures:
 *   - BST (std::map<length, vector<FragRef*>>): O(log N) best-fit lookup
 *   - FIFO (std::deque<FragRef*>): insertion-order queue for oldest-first
 *   - FragRef with `consumed` flag for lazy FIFO deletion
 *
 * Following Megatron-LM's pybind11 pattern (helpers.cpp).
 *
 * Compile: make -C titan_sci/datasets/dataloader/
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <list>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// ── Fragment reference ──────────────────────────────────────────────────────

// Forward decl so FragRef can hold an iterator into FragBuffer's FIFO list
// (which is a list of FragRef*).
struct FragRef;
using FragFifo = std::list<FragRef*>;

struct FragRef {
    int32_t chunk_order_idx;
    int32_t eff_doc_idx;
    int64_t token_start;
    int64_t token_end;
    // Iterator pointing at this FragRef's slot in the FIFO list. Stored so
    // detach() can erase the entry in O(1) without scanning the list.
    FragFifo::iterator fifo_it;

    int64_t length() const { return token_end - token_start; }
};

// ── Dynamic bitset using vector<uint64_t> ───────────────────────────────────

class DynBitset {
public:
    explicit DynBitset(int64_t num_bits)
        : n_bits_(num_bits),
          n_words_((num_bits + 63) / 64),
          data_(n_words_, 0ULL) {}

    void zero() { std::fill(data_.begin(), data_.end(), 0ULL); }

    void set(int64_t pos) {
        if (pos >= 0 && pos < n_bits_)
            data_[pos / 64] |= (1ULL << (pos % 64));
    }

    bool test(int64_t pos) const {
        if (pos < 0 || pos >= n_bits_) return false;
        return (data_[pos / 64] >> (pos % 64)) & 1ULL;
    }

    // Find highest set bit <= max_pos. Returns -1 if none.
    int64_t highest_set_leq(int64_t max_pos) const {
        if (max_pos < 0) return -1;
        if (max_pos >= n_bits_) max_pos = n_bits_ - 1;
        int64_t word_idx = max_pos / 64;
        // Check partial word
        uint64_t mask = data_[word_idx] & ((max_pos % 64 == 63)
                        ? ~0ULL : ((1ULL << ((max_pos % 64) + 1)) - 1));
        if (mask) return word_idx * 64 + 63 - __builtin_clzll(mask);
        for (int64_t w = word_idx - 1; w >= 0; --w) {
            if (data_[w]) return w * 64 + 63 - __builtin_clzll(data_[w]);
        }
        return -1;
    }

    // OR-assign shifted version: this |= (src << shift)
    void or_shifted(const DynBitset &src, int64_t shift) {
        if (shift <= 0 || shift >= n_bits_) {
            if (shift == 0) {
                for (int64_t i = 0; i < n_words_; ++i)
                    data_[i] |= src.data_[i];
            }
            return;
        }
        int64_t word_shift = shift / 64;
        int bit_shift = static_cast<int>(shift % 64);

        if (bit_shift == 0) {
            for (int64_t i = n_words_ - 1; i >= word_shift; --i)
                data_[i] |= src.data_[i - word_shift];
        } else {
            for (int64_t i = n_words_ - 1; i >= word_shift; --i) {
                uint64_t val = src.data_[i - word_shift] << bit_shift;
                if (i - word_shift > 0)
                    val |= src.data_[i - word_shift - 1] >> (64 - bit_shift);
                data_[i] |= val;
            }
        }
    }

    // Compute dst = src | (src << shift) in a single pass.
    // Replaces the assign(src) + or_shifted(src, shift) pair used by the DP,
    // halving the per-step word traffic and removing dst zero-init.
    static void copy_shifted_or(const DynBitset &src, DynBitset &dst,
                                int64_t shift) {
        const int64_t n = src.n_words_;
        if (shift <= 0 || shift >= src.n_bits_) {
            if (shift == 0) {
                for (int64_t i = 0; i < n; ++i)
                    dst.data_[i] = src.data_[i] | src.data_[i];
            } else {
                for (int64_t i = 0; i < n; ++i)
                    dst.data_[i] = src.data_[i];
            }
            return;
        }
        int64_t word_shift = shift / 64;
        int bit_shift = static_cast<int>(shift % 64);

        if (bit_shift == 0) {
            for (int64_t i = n - 1; i >= 0; --i) {
                uint64_t shifted = (i >= word_shift) ? src.data_[i - word_shift] : 0ULL;
                dst.data_[i] = src.data_[i] | shifted;
            }
        } else {
            for (int64_t i = n - 1; i >= 0; --i) {
                uint64_t shifted = 0ULL;
                if (i >= word_shift) {
                    shifted = src.data_[i - word_shift] << bit_shift;
                    if (i - word_shift > 0)
                        shifted |= src.data_[i - word_shift - 1] >> (64 - bit_shift);
                }
                dst.data_[i] = src.data_[i] | shifted;
            }
        }
    }

    void assign(const DynBitset &other) {
        std::copy(other.data_.begin(), other.data_.end(), data_.begin());
    }

private:
    int64_t n_bits_;
    int64_t n_words_;
    std::vector<uint64_t> data_;
};

// ── Tree + FIFO buffer ──────────────────────────────────────────────────────
//
// Lifetime contract (changed from a previous lazy-deletion design):
//   * FragBuffer does NOT own its FragRefs. add() new's a FragRef and tracks
//     it in the BST + FIFO list. pop_oldest() and detach() both REMOVE the
//     FragRef from those structures; the caller becomes the sole owner and
//     must delete it once it's done with it.
//   * The previous version kept every FragRef alive in a `storage` vector
//     until the build finished. That OOM'd on multi-T-token datasets
//     (~70 GB+ for 1.5B fragments) even after we streamed the output.
//   * std::list<FragRef*> for the FIFO + an iterator stored in each FragRef
//     gives O(1) erase from anywhere in the FIFO, which is what makes
//     detach() cheap enough to skip lazy deletion entirely.

class FragBuffer {
public:
    using FragPtr = FragRef*;

    // BST: length → list of fragment pointers
    std::map<int64_t, std::vector<FragPtr>> bst;
    // FIFO list — only contains live fragments. No zombies.
    FragFifo fifo;

    int64_t live_count = 0;  // non-consumed fragments

    FragPtr add(int32_t chunk_idx, int32_t doc_idx,
                int64_t start, int64_t end) {
        FragPtr ptr = new FragRef{chunk_idx, doc_idx, start, end, fifo.end()};
        int64_t len = ptr->length();
        bst[len].push_back(ptr);
        fifo.push_back(ptr);
        // After push_back, set the iterator to the just-inserted element
        // (list iterators are stable across other pushes/erases, so this
        // remains valid until the entry is erased).
        auto it = fifo.end();
        --it;
        ptr->fifo_it = it;
        live_count++;
        return ptr;
    }

    // Pop oldest fragment. Removes from BST + FIFO. Caller owns the returned
    // pointer (must delete it).
    FragPtr pop_oldest() {
        if (fifo.empty()) return nullptr;
        FragPtr f = fifo.front();
        fifo.pop_front();
        remove_from_bst(f);
        live_count--;
        return f;
    }

    // Detach a specific fragment from BST + FIFO. Caller owns the pointer
    // (must delete it).
    void detach(FragPtr f) {
        remove_from_bst(f);
        fifo.erase(f->fifo_it);
        live_count--;
    }

    // Append all live fragments to caller-owned buffer (no allocation).
    void collect_live_into(std::vector<FragPtr> &out) const {
        for (const auto &pair : bst) {
            for (FragPtr f : pair.second) {
                out.push_back(f);
            }
        }
    }

    // Drain everything still in the buffer; deletes each FragRef. Used only
    // for cleanup on early exit so we don't leak.
    ~FragBuffer() {
        for (FragPtr f : fifo) delete f;
        fifo.clear();
        bst.clear();
        live_count = 0;
    }

private:
    void remove_from_bst(FragPtr f) {
        int64_t len = f->length();
        auto it = bst.find(len);
        if (it == bst.end()) return;
        auto &vec = it->second;
        for (auto vit = vec.begin(); vit != vec.end(); ++vit) {
            if (*vit == f) {
                vec.erase(vit);
                break;
            }
        }
        if (vec.empty()) bst.erase(it);
    }
};

// ── RAII wrapper for buffered file output ──────────────────────────────────

class BufferedFile {
public:
    BufferedFile(const std::string &path, size_t buf_size)
        : path_(path), buf_(buf_size) {
        f_ = std::fopen(path.c_str(), "wb");
        if (!f_) {
            throw std::runtime_error(
                "bestfit_packer: failed to open '" + path + "' for writing: "
                + std::strerror(errno));
        }
        // _IOFBF = full buffering. The buffer must outlive the FILE*, which it
        // does (owned by this object).
        std::setvbuf(f_, buf_.data(), _IOFBF, buf_.size());
    }

    ~BufferedFile() {
        if (f_) std::fclose(f_);
    }

    BufferedFile(const BufferedFile &) = delete;
    BufferedFile &operator=(const BufferedFile &) = delete;

    void write(const void *data, size_t size) {
        size_t n = std::fwrite(data, 1, size, f_);
        if (n != size) {
            throw std::runtime_error(
                "bestfit_packer: short write to '" + path_ + "' ("
                + std::to_string(n) + "/" + std::to_string(size) + " bytes): "
                + std::strerror(errno));
        }
    }

    void close() {
        if (f_) {
            int rc = std::fclose(f_);
            f_ = nullptr;
            if (rc != 0) {
                throw std::runtime_error(
                    "bestfit_packer: fclose('" + path_ + "') failed: "
                    + std::strerror(errno));
            }
        }
    }

private:
    std::string path_;
    std::FILE *f_ = nullptr;
    std::vector<char> buf_;
};

// ── Main packing function ───────────────────────────────────────────────────
//
// Streams the packing plan to disk so the packer's resident memory stays
// bounded (≈ 32 MB write buffers + the BST/FIFO buffer) regardless of
// dataset size. Without this, doc_refs_flat grew to 100s of GB for
// multi-T-token datasets and got OOM-killed on the login node.

std::tuple<int64_t, int64_t> build_bestfit_plan(
    const py::array_t<int64_t> &doc_lengths_,
    const py::array_t<int32_t> &chunk_ids_,
    const py::array_t<int32_t> &doc_indices_,
    const int32_t tokens_per_seq,
    const int32_t min_seq_len,
    const int32_t tree_size,
    const int64_t seed,
    const std::string &counts_path,
    const std::string &refs_path)
{
    (void) seed;  // kept for API compatibility — algorithm is deterministic

    auto doc_lengths = doc_lengths_.unchecked<1>();
    auto chunk_ids = chunk_ids_.unchecked<1>();
    auto doc_indices = doc_indices_.unchecked<1>();
    const int64_t num_docs = doc_lengths_.shape(0);

    // 16 MB write buffers — large enough that fwrite is essentially memcpy
    // until the buffer fills, at which point one big block hits disk.
    constexpr size_t WRITE_BUF = 16 * 1024 * 1024;
    BufferedFile counts_file(counts_path, WRITE_BUF);
    BufferedFile refs_file(refs_path, WRITE_BUF);

    int64_t num_sequences = 0;
    int64_t num_refs = 0;

    FragBuffer buffer;
    int64_t doc_cursor = 0;

    // Reusable scratch for live_fragments() — avoids allocating a fresh
    // std::vector per emit (called ~total_sequences times).
    std::vector<FragRef*> live_scratch;
    live_scratch.reserve(static_cast<size_t>(tree_size) + 8);

    // Pre-allocate the DP bitset stack once. Each emit reuses the same
    // (tree_size + 2) bitsets, just rewriting their bits instead of
    // re-allocating ~tree_size DynBitsets (each owning a heap vector) per
    // sequence. With tree_size=100 and ~85M sequences this removes ~8.5B
    // heap allocations.
    const int dp_capacity = tree_size + 2;
    std::vector<DynBitset> dp_pool;
    dp_pool.reserve(dp_capacity);
    for (int i = 0; i < dp_capacity; ++i)
        dp_pool.emplace_back(static_cast<int64_t>(tokens_per_seq) + 1);

    // Progress logging — every PROGRESS_EVERY emitted sequences, print a
    // brief line to stderr so a multi-minute build is observable.
    constexpr int64_t PROGRESS_EVERY = 1000000;
    int64_t next_progress_at = PROGRESS_EVERY;
    const auto t_start = std::chrono::steady_clock::now();

    // Helper: emit a sequence (list of fragment refs) — streams to disk.
    auto emit_sequence = [&](const std::vector<FragRef*> &refs) {
        int64_t total = 0;
        for (const auto *f : refs) total += f->length();
        if (total < min_seq_len) return;

        const int32_t count = static_cast<int32_t>(refs.size());
        counts_file.write(&count, sizeof(int32_t));
        num_sequences++;

        for (const auto *f : refs) {
            const int64_t r[4] = {
                static_cast<int64_t>(f->chunk_order_idx),
                static_cast<int64_t>(f->eff_doc_idx),
                f->token_start,
                f->token_end,
            };
            refs_file.write(r, sizeof(r));
            num_refs++;
        }
    };

    // Helper: emit a single full-length sequence directly (no tree involvement)
    auto emit_full_seq = [&](int32_t chunk_id, int32_t doc_idx,
                             int64_t start, int64_t end) {
        const int32_t count = 1;
        counts_file.write(&count, sizeof(int32_t));
        num_sequences++;
        const int64_t r[4] = {
            static_cast<int64_t>(chunk_id),
            static_cast<int64_t>(doc_idx),
            start,
            end,
        };
        refs_file.write(r, sizeof(r));
        num_refs++;
    };

    // Helper: add one document to the tree, emitting full-sequence chunks
    auto add_doc = [&](int64_t eff_len, int32_t chunk_id, int32_t doc_idx) {
        if (eff_len < min_seq_len) return;

        int64_t offset = 0;

        // Extract full-sequence chunks (doc >= 2 * tokens_per_seq)
        while (eff_len - offset >= 2 * static_cast<int64_t>(tokens_per_seq)) {
            emit_full_seq(chunk_id, doc_idx, offset, offset + tokens_per_seq);
            offset += tokens_per_seq;
        }

        int64_t remainder = eff_len - offset;

        if (remainder >= tokens_per_seq) {
            // Between 1x and 2x tokens_per_seq: split in half
            int64_t half = remainder / 2;
            int64_t other_half = remainder - half;
            if (half >= min_seq_len)
                buffer.add(chunk_id, doc_idx, offset, offset + half);
            if (other_half >= min_seq_len)
                buffer.add(chunk_id, doc_idx, offset + half, offset + remainder);
        } else if (remainder >= min_seq_len) {
            // Short fragment: insert directly
            buffer.add(chunk_id, doc_idx, offset, offset + remainder);
        }
    };

    // Reusable container for the FragRefs that compose one emitted sequence.
    // Cleared (not re-allocated) at the start of each emit. Frags are deleted
    // at the end of each emit so the buffer's net memory use stays bounded
    // to the tree_size live fragments plus a handful in flight.
    std::vector<FragRef*> sequence_frags;
    sequence_frags.reserve(static_cast<size_t>(tree_size) + 4);

    auto delete_sequence_frags = [&]() {
        for (FragRef *f : sequence_frags) delete f;
        sequence_frags.clear();
    };

    // Helper: emit one sequence from the tree using oldest-first + subset-sum DP
    auto emit_one_from_tree = [&]() -> bool {
        if (buffer.live_count == 0) return false;

        // Step a: pop oldest fragment as first pick (caller owns it)
        FragRef *first = buffer.pop_oldest();
        if (!first) return false;

        int64_t remaining = tokens_per_seq - first->length();
        sequence_frags.push_back(first);

        if (remaining <= 0) {
            // First fragment fills entire sequence (shouldn't happen normally,
            // but handle gracefully)
            emit_sequence(sequence_frags);
            delete_sequence_frags();
            return true;
        }

        if (buffer.live_count == 0) {
            // No other fragments available — emit as partial
            emit_sequence(sequence_frags);
            delete_sequence_frags();
            return true;
        }

        // Step b: collect live fragments for DP (reuse scratch vector)
        live_scratch.clear();
        buffer.collect_live_into(live_scratch);
        const int N = static_cast<int>(live_scratch.size());

        // Collect lengths; only fragments with length <= R participate in DP
        const int64_t R = remaining;  // target sum

        // Only include fragments that fit wholly (length <= R). We index
        // live_scratch directly via dp_indices instead of building a
        // separate `lengths` array.
        static thread_local std::vector<int> dp_indices;
        dp_indices.clear();
        dp_indices.reserve(N);
        for (int i = 0; i < N; ++i) {
            if (live_scratch[i]->length() <= R)
                dp_indices.push_back(i);
        }
        const int M = static_cast<int>(dp_indices.size());

        // Forward pass — reuse pre-allocated dp_pool (no per-emit alloc)
        if (M + 1 > static_cast<int>(dp_pool.size())) {
            // Defensive: should not happen given dp_capacity = tree_size + 2,
            // but grow if a caller ever passes more fragments than expected.
            for (int i = static_cast<int>(dp_pool.size()); i < M + 1; ++i)
                dp_pool.emplace_back(static_cast<int64_t>(tokens_per_seq) + 1);
        }
        dp_pool[0].zero();
        dp_pool[0].set(0);

        for (int i = 0; i < M; ++i) {
            // dp[i+1] = dp[i] | (dp[i] << shift) — single-pass.
            DynBitset::copy_shifted_or(
                dp_pool[i], dp_pool[i + 1],
                live_scratch[dp_indices[i]]->length());
        }

        // Find best reachable sum ≤ R
        int64_t best = dp_pool[M].highest_set_leq(R);
        if (best < 0) best = 0;

        // Backward trace: identify selected fragments
        static thread_local std::vector<int> selected_dp_idx;
        selected_dp_idx.clear();
        {
            int64_t target = best;
            for (int i = M - 1; i >= 0 && target > 0; --i) {
                int64_t frag_len = live_scratch[dp_indices[i]]->length();
                if (target >= frag_len && dp_pool[i].test(target - frag_len)) {
                    selected_dp_idx.push_back(i);
                    target -= frag_len;
                }
            }
        }

        // Detach selected fragments from buffer, add to sequence_frags. Each
        // detach is O(1) thanks to the FIFO list iterator stored on FragRef.
        for (int di : selected_dp_idx) {
            FragRef *f = live_scratch[dp_indices[di]];
            buffer.detach(f);
            sequence_frags.push_back(f);
        }

        // Any remaining gap (R - best > 0) becomes padding, handled by collator.

        emit_sequence(sequence_frags);
        delete_sequence_frags();
        return true;
    };

    // ── Main loop ───────────────────────────────────────────────────────

    while (doc_cursor < num_docs || buffer.live_count > 0) {
        // Fill phase: add documents until tree reaches tree_size
        while (buffer.live_count < tree_size && doc_cursor < num_docs) {
            add_doc(doc_lengths[doc_cursor], chunk_ids[doc_cursor],
                    doc_indices[doc_cursor]);
            doc_cursor++;
        }

        // Emit phase: compose sequences from tree
        if (buffer.live_count > 0) {
            if (!emit_one_from_tree()) break;
        }

        // If tree dropped below tree_size and docs remain, loop to refill
        // Otherwise continue emitting
        if (buffer.live_count >= tree_size || doc_cursor >= num_docs) {
            // Keep emitting until we need to refill
            while (buffer.live_count >= tree_size ||
                   (doc_cursor >= num_docs && buffer.live_count > 0)) {
                if (!emit_one_from_tree()) break;
                // If docs remain and tree dropped below threshold, break to refill
                if (buffer.live_count < tree_size && doc_cursor < num_docs) break;
            }
        }

        // Periodic progress output (stderr, line-buffered) so a multi-minute
        // build doesn't look hung. Skips counting empty emits (returned false).
        const int64_t emitted = num_sequences;
        if (emitted >= next_progress_at) {
            const auto now = std::chrono::steady_clock::now();
            const double elapsed =
                std::chrono::duration<double>(now - t_start).count();
            const double doc_frac =
                num_docs > 0
                    ? static_cast<double>(doc_cursor) / static_cast<double>(num_docs)
                    : 1.0;
            std::fprintf(stderr,
                "[bestfit_packer] %.1fs elapsed, emitted=%lld seqs, "
                "doc_cursor=%lld/%lld (%.1f%%), buffer=%lld\n",
                elapsed, static_cast<long long>(emitted),
                static_cast<long long>(doc_cursor),
                static_cast<long long>(num_docs),
                100.0 * doc_frac,
                static_cast<long long>(buffer.live_count));
            std::fflush(stderr);
            next_progress_at += PROGRESS_EVERY;
        }
    }

    // Flush + close the output streams; surface any I/O error to Python.
    counts_file.close();
    refs_file.close();

    return std::make_tuple(num_sequences, num_refs);
}

PYBIND11_MODULE(bestfit_packer, m) {
    m.doc() = "Best-fit packing plan builder for BestFitPackedDataset";

    m.def("build_bestfit_plan", &build_bestfit_plan,
          py::arg("doc_lengths"),
          py::arg("chunk_ids"),
          py::arg("doc_indices"),
          py::arg("tokens_per_seq"),
          py::arg("min_seq_len"),
          py::arg("buffer_size") = 500,
          py::arg("seed") = 1,
          py::arg("counts_path"),
          py::arg("refs_path"),
          R"doc(
Build a best-fit packing plan, streaming output to disk.

The packing plan is written as raw binary files:
  - counts_path: int32 little-endian, one per packed sequence (== fragments
                 in that sequence)
  - refs_path:   int64 little-endian, four ints per fragment in row-major
                 order: (chunk_order_idx, eff_doc_idx, token_start, token_end)

Streaming the output bounds resident memory to ~32 MB (write buffers + the
BST/FIFO state). The previous in-memory accumulator grew to 100s of GB on
multi-T-token corpora and got OOM-killed.

Algorithm: fills a BST buffer (tree) to tree_size fragments, then emits
sequences by:
1. Popping the oldest fragment (FIFO) as first pick (prevents starvation)
2. Subset-sum DP to find optimal subset filling remaining space
3. Any remaining gap is left as padding (handled by collator)

Args:
    doc_lengths: int64[N] - effective document lengths (with EOS) in
                 permuted chunk order
    chunk_ids:   int32[N] - chunk_order_idx for each document
    doc_indices: int32[N] - effective doc index within chunk for each doc
    tokens_per_seq: target tokens per sequence (seq_len + 1)
    min_seq_len: minimum sequence/fragment length to keep
    buffer_size: tree capacity before emitting (default 500)
    seed:        unused (kept for API compatibility)
    counts_path: where to write the int32 counts file
    refs_path:   where to write the int64[N, 4] refs file

Returns:
    tuple (num_sequences, num_refs) — sizes of the two written files in
    elements (not bytes). Used by the Python wrapper to set up np.memmap.
)doc");
}
