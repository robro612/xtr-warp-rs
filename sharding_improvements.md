# Sharding Improvements Roadmap

This document tracks sharding-related memory/performance work completed so far, plus deferred next steps.

## Completed (current branch)

- Sharded compaction switched from all-shards-at-once allocation to one-shard-at-a-time allocation.
  - Effect: compaction-local peak memory scales approximately with `1 / num_shards`.
- Sharded search path uses per-search `SearchConfig` (matching unsharded behavior).
- Profiling tests added for build/retrieval memory trends and phase-split reporting.
- Encode profiling added:
  - `XTR_WARP_PROFILE_ENCODE=1` for active batch knobs
  - `XTR_WARP_PROFILE_ENCODE_LOCAL=1` for local tensor footprint estimates
- Encode microbatching improved and shard-aware batch-size scaling added for hot paths.
- Minimal streaming fix added in encode:
  - per-microbatch `code_batch` and packed residuals are moved to CPU immediately
  - avoids retaining full chunk output tensors on GPU before write
- Added Rust codec-sample memory cap:
  - `XTR_WARP_CODEC_SAMPLE_CAP=<int>`
  - caps the create-time sample used for residual codec training.

## Current observed behavior

- Compaction-local memory scales close to ideal `1/s`.
- Encode-local working-set estimates also scale close to `1/s`.
- Whole-create peak follows `baseline + variable/s`, where some non-shard-scaled terms remain.
- With uncapped codec sample, create peak was still large at realistic synthetic scale.
- With codec sample cap enabled, create peak dropped substantially while preserving shard scaling.

## Shard loading and mutability notes

- Core mutable operations now run on sharded indices as well:
  - `delete` (tombstones + search-time filtering)
  - `add` (incremental merge into sharded compacted structures)
  - `update` (replace-in-place semantics with stable passage IDs)
  - `compact` (rewrites chunks, rebuilds sharded compacted files, clears tombstones)
- `load()` behavior for sharded indexes is intentionally strict:
  - passing a single device replicates it to all shards
  - passing a list requires exactly `num_shards` devices
  - any other device-count mismatch raises a `ValueError`
- Current limitation: there is no automatic shard rebalancing after index growth
  (e.g., centroid expansion can extend the last shard boundary). Correctness is
  preserved, but long-lived indexes may drift in per-shard load balance.

## Chunk files vs sharded compacted layout (disk and incrementals)

- **Search** only needs root metadata (e.g. `centroids.npy`, `bucket_weights.npy`) plus each `shard_*` directory’s `sizes.compacted.npy`, `codes.compacted.npy` (passage IDs per embedding row, despite the name), and `residuals.compacted.npy`. The per-chunk `*.codes.npy`, `*.residuals.npy`, `doclens.*`, `*.passage_ids.npy`, etc. are not read by the sharded loader.
- **Incremental add/remove** is workable on the compacted sharded representation: this repo already merges updates into `shard_*/…compacted.npy` (and uses tombstones for delete). Uncompacted chunk tensors are the current encode/compaction *checkpoint*, not a separate source of truth—you could refactor pipelines to append or filter compacted shards directly instead of round-tripping through chunk `.npy` files.
- For a **read-only** index, deleting the chunk-layer copies after build removes a near-duplicate of the bulk tensor data (packed residuals and parallel per-row metadata). In practice that often **roughly halves** on-disk index footprint, aside from small shared files.

## Profiling findings (A/B snapshots)

### Dataset profile used

- `num_docs=50_000`
- `doc_len=1024`
- `dim=128`
- `nbits=4`
- KMeans sample (Python) set to `1024`

### Baseline (no codec cap)

- `create_peak_mb`:
  - 1 shard: `19581.0`
  - 2 shards: `17824.3`
  - 4 shards: `15869.4`
- Encode-local (`max_working_set_est_mb`) scaled ~`1/s`:
  - `6092.5 -> 3046.2 -> 1523.1`
- Compaction-local scaled ~`1/s`:
  - mono-equivalent `3515.6`
  - sharded peak `1758.1` (2 shards), `879.1` (4 shards)

### Treatment (`XTR_WARP_CODEC_SAMPLE_CAP=1024`)

- `create_peak_mb`:
  - 1 shard: `10039.0`
  - 2 shards: `8282.3`
  - 4 shards: `6327.4`
- Absolute reductions vs baseline:
  - 1 shard: `-48.7%`
  - 2 shards: `-53.5%`
  - 4 shards: `-60.1%`
- Encode-local and compaction-local scaling remained intact.

### Interpretation

- A major non-shard-scaled create-time contributor was the Rust codec sampling/training path.
- Capping codec sample size dramatically reduced whole-create peak.
- The remaining deviation from perfect `1/s` appears to be process/runtime/allocator baseline plus other non-shard-scaled terms.

### Practical takeaway

- For large corpora, use a codec-sample cap while evaluating memory headroom.
- Keep in mind quality tradeoffs; tune cap empirically on retrieval metrics.

## Deferred next steps

### 1) Fully shard-routed encode outputs (recommended next)

Current minimal streaming still writes chunk outputs first, then compaction routes by centroid range.

Next version:

- During encode, route each microbatch row directly by centroid -> shard.
- Append to per-shard writers/buffers immediately.
- Finalize per-shard files directly, minimizing downstream routing work.

Expected impact:

- Further reduce non-sharded encode residency.
- Improve end-to-end memory scaling toward `1/s`.

### 2) Avoid large per-batch reconstruction temporaries

Potential improvements:

- Fuse/restructure residual packing path to reduce peak live intermediates
- Reduce or tile `recon_centroids`/residual transform buffers
- More aggressive control of score/code batch dimensions based on shard count and device memory budget

### 3) Parallel shard encode workers (without all-reduce)

Near-term parallel option that avoids all-reduce complexity and can improve throughput
while preserving memory gains. Partitioning is by centroid range, matching the on-disk
shard layout and search-time loading — so encode output lands directly in the right shard.

#### Prerequisites (already implemented)

Before parallel encoding begins, two single-device steps produce shared artifacts:

1. **K-means** → centroids matrix (Python side, `compute_kmeans` in `search.py`,
   on sampled embeddings).
2. **Codec training** → bucket_cutoffs, bucket_weights, avg_residual
   (`train_residual_codec` in `create.rs`, on sampled embeddings controlled by
   `codec_sample_cap`).

Both operate on a small sample, not the full corpus. The centroids and codec are then
replicated to each device before the parallel encode begins.

#### Encoding flow

**Pass 1 — Centroid assignment (CPU coordinator + N GPUs):**

1. Stream raw embeddings from `EmbeddingSource` on CPU in microbatches.
2. Split each microbatch into N chunks, send chunk i to GPU i.
3. Each GPU computes `chunk @ centroids.T` → argmax (centroid codes).
4. Gather the codes (small int tensors) back to CPU.
5. CPU accumulates the full code vector for all embeddings.

If the centroid assignment matmul is not a bottleneck, steps 2-4 can run on a single
device instead. The matmul grows with `num_embeddings × num_centroids`, but it is
embarrassingly parallel over the embedding dimension and can be chunked across GPUs
if needed.

**Pass 2 — Residual encoding (CPU coordinator dispatches to N shard workers):**

1. Re-read embeddings from `EmbeddingSource` (two-pass avoids holding full corpus
   in CPU memory; for in-memory sources the re-read is free).
2. Pair each embedding with its centroid assignment from Pass 1.
3. Partition microbatches by centroid range → shard buckets.
4. Send each shard's embedding slice to that shard's GPU.
5. On each GPU (in parallel): compute residuals against local centroids copy,
   bucketize, bit-pack, and write (append) to per-shard chunk files on disk.
6. Free GPU memory — nothing accumulates across microbatches.

Each shard writes to its own directory (`shard_0/`, `shard_1/`, ...) with its own
chunk index counter. No write conflicts, no locks, no coordination after dispatch.

**Pass 3 — Per-shard compaction:**

After all microbatches are encoded, each shard independently compacts its chunk files
into the final centroid-sorted layout (`sizes.compacted.npy`, `codes.compacted.npy`,
`residuals.compacted.npy`). This is the same counting-sort compaction as today, just
operating on 1/N of the data. Can also be parallelized across shards.

#### Per-device memory profile

At any point during encoding, each GPU holds:

- **Persistent**: centroids `[num_centroids, dim]` f16 + codec tensors (small)
- **Transient (Pass 1)**: `[microbatch_size/N, dim]` embeddings +
  `[microbatch_size/N, num_centroids]` score matrix (freed after argmax)
- **Transient (Pass 2)**: ~`[microbatch_size/N, dim]` shard slice +
  residual/packing intermediates (freed after writing chunk files)

Nothing accumulates across microbatches. Peak per-device memory is essentially the
same as today's single-GPU encode working set, but with ~1/N of the data per batch.

#### Key implementation changes

| File | Change |
|------|--------|
| `rust/utils/residual_codec.rs` | Add `to_device()` to deep-copy codec to another GPU |
| `rust/index/source.rs` | Support two-pass iteration (re-read after assignment pass) |
| `rust/index/create.rs` | New `encode_chunks_parallel()` orchestrator: run Pass 1, spawn N threads for Pass 2, merge results |
| `rust/index/encode.rs` | No structural changes — already takes all deps as args |
| `rust/lib.rs` | Add `encoding_devices: Option<Vec<String>>` to PyO3 `create()` |
| `python/xtr_warp/search.py` | Thread `encoding_devices` parameter through |

Thread safety: `std::thread::scope` with one thread per GPU, following the existing
pattern in `scorer.rs` for sharded search. Each thread owns its device-local codec
and centroids (deep-copied, not shared).

#### Code path split: single-device vs. multi-device

The two-pass flow (assign centroids, then re-read and encode) is unnecessary overhead
for the single-device case, where centroid assignment and residual encoding can be fused
in a single pass — which is exactly what `encode_chunks_inner` does today.

Rather than accepting the overhead or threading multi-device concerns into the
single-device loop, the split should be at the orchestration level in `create_index()`:

- `encode_chunks()` — existing single-pass, single-device path (unchanged)
- `encode_chunks_parallel()` — new two-pass, multi-device path

Both call the same `encode_embedding_batch()` for the actual GPU-side residual/bucketize/
bit-pack work. Both write the same chunk file format. Both feed into the same compaction
step. The only difference is the loop structure (fused vs. two-pass with dispatch).

```rust
if encoding_devices.len() > 1 {
    encode_chunks_parallel(...)
} else {
    encode_chunks(...)  // existing path, unchanged
}
```

#### Verification tests for parallel encode

1. **Exact chunk equivalence**: Create the same index with `encode_chunks()` (single
   device) and `encode_chunks_parallel()` (N devices) using the same seed. Chunk files
   should be bitwise identical — same codes, same packed residuals, same doclens.

2. **Exact search equivalence**: Search a parallel-created sharded index and a
   single-device-created sharded index with identical queries. PIDs and scores must
   match exactly (not approximately).

3. **N=1 multi-device path**: Run `encode_chunks_parallel()` with a single device.
   Verify output is identical to `encode_chunks()`. Catches edge cases in the two-pass
   flow when there's no actual partitioning.

4. **Shard file isolation**: Verify each shard directory contains only chunk files for
   centroids in its range — no centroid codes outside `[centroid_start, centroid_end)`
   in any shard's chunk files.

5. **Global centroid counts**: Verify the merged centroid counts from all shard workers
   sum to the same counts produced by the single-device path.

6. **Disk source two-pass**: Test specifically with `DiskEmbeddingSource` to verify
   the re-read in Pass 2 produces the same embeddings in the same order as Pass 1.
   In-memory sources get this for free.

### 4) Multi-GPU all-to-all / collective encode (longer-term)

Ambitious option beyond step 3:

- Distribute assignment/packing across multiple GPUs.
- All-to-all route rows to owning shard writers.

Risks:

- Higher implementation complexity.
- Communication overhead may offset compute gains without careful design.

### 5) Stronger phase isolation profiling

For more definitive attribution:

- Run create phases in separate subprocesses (fresh allocator state).
- Capture per-phase NVML process memory over time.
- Keep these as opt-in profiling tools (not default CI tests).

### 6) Codec sample step scalability

Current behavior:

- Codec training sample can be memory-heavy if uncapped.
- This is now mitigated by `codec_sample_cap` / `XTR_WARP_CODEC_SAMPLE_CAP`.

Options:

1. **Single-device capped sample (current default path)**
   - Keep codec sampling on one device.
   - Use cap to bound memory.
   - Lowest complexity; good near-term default.

2. **Streaming CPU statistics/sketches (recommended next)**
   - Compute residual distribution statistics incrementally (hist/sketch/tdigest-style).
   - Avoid building a large GPU sample tensor.
   - Medium complexity; avoids multi-GPU comms.

3. **Distributed multi-GPU codec sampling**
   - Shard sample/stat computation across GPUs with collective aggregation.
   - Highest complexity (communication + synchronization).
   - Potentially best throughput at very large scale.

Recommendation:

- Keep (1) for now, prototype (2) when needed, reserve (3) for large-scale throughput-driven scenarios.

### 7) Reuse k-means sample for codec training

Currently, `compute_kmeans` (Python) and `plan_and_sample` (Rust) each perform an
independent full disk scan to sample embeddings — same formula (`16*sqrt(120*N)`),
same distribution, but two separate passes over all embedding files. For large corpora
(e.g. 8.8M docs) this adds several minutes of redundant I/O.

Proposal: pass the k-means sample tensor and `total_doc_len` from Python into Rust's
`create_index`, skipping the Rust-side sampling entirely.

Changes:
- `compute_kmeans` returns `(centroids, dim, sampled_embeddings, total_doc_len)`
- `xtr_warp_rs.create()` accepts optional `codec_embeddings: PyTensor` and `total_doc_len: i64`
- `plan_and_sample` skips disk scan when pre-sampled data is provided

Statistical note: the k-means sample trains the centroids, so its residuals may be
slightly optimistic vs. held-out data. In practice the bias is negligible — k-means
internally subsamples to `k * max_points_per_centroid` tokens, so most sampled tokens
never directly influenced the centroids. The codec's existing 5% heldout split
(in `train_residual_codec`) further mitigates this. If needed, the heldout portion
could be drawn from tokens beyond the k-means subsampling threshold for zero bias.

## Useful env knobs

- `XTR_WARP_PROFILE_COMPACTION=1`
- `XTR_WARP_PROFILE_ENCODE=1`
- `XTR_WARP_PROFILE_ENCODE_LOCAL=1`
- `XTR_WARP_ENCODE_EMB_BATCH_SIZE=<int>`
- `XTR_WARP_ENCODE_SCORE_BATCH_SIZE=<int>`
- `XTR_WARP_ENCODE_CODE_BATCH_SIZE=<int>`
- `XTR_WARP_MAX_SCORE_ELEMS=<int>`
- `XTR_WARP_CODEC_SAMPLE_CAP=<int>`

