# Multi-GPU Sharded Index for WARP

## Context

The current implementation requires the entire index (compacted PIDs, residuals, offsets, sizes) to fit on a single device. For large collections this is a hard wall. The goal is to partition the index across N GPUs by **centroid range** — the natural shard boundary given the centroid-sorted compacted layout — so that collections can scale beyond a single device's memory.

Design principle: **each GPU decompresses its shard's centroids; a CPU coordinator runs the merge on the reassembled cells — producing results identical to the single-shard case.** No GPU-to-GPU transfers (tch-rs has no NCCL). Single-shard (N=1) behaves identically to the current code. Multi-shard produces **exact** same results as single-shard (not an approximation).

---

## 1. New/Modified Data Structures (`rust/utils/types.rs`)

**New structs:**

```rust
pub struct ShardConfig {
    pub shard_id: usize,
    pub num_shards: usize,
    pub centroid_start: usize,  // inclusive
    pub centroid_end: usize,    // exclusive
    pub device: Device,
}

pub struct ShardedIndex {
    pub shards: Vec<Arc<ReadOnlyIndex>>,
    pub shard_configs: Vec<ShardConfig>,
    pub metadata: IndexMetadata,
}
```

Each `ReadOnlyIndex` in `shards` contains:
- **Replicated**: full `centroids` [num_centroids, dim], `bucket_weights` (small — ~16-256MB)
- **Local**: this shard's slice of `sizes/pids/residuals/offsets_compacted`

**`IndexMetadata` additions:**
- `num_shards: Option<usize>` — absent or `1` for legacy indices
- `shard_boundaries: Option<Vec<usize>>` — `[0, b1, b2, ..., num_centroids]`

**`LoadedIndex` addition:**
- `shard_config: Option<ShardConfig>` — used by the decompressor for centroid ID translation

---

## 2. On-Disk Format

```
index_dir/
  metadata.json              # now includes num_shards, shard_boundaries
  centroids.npy              # full, replicated on load
  bucket_weights.npy         # full, replicated on load
  bucket_cutoffs.npy         # unchanged
  avg_residual.npy           # unchanged
  shard_0/
    sizes.compacted.npy      # [centroid_end_0 - centroid_start_0]
    codes.compacted.npy      # [shard_0_total_embeddings]
    residuals.compacted.npy  # [shard_0_total_embeddings, packed_dim]
  shard_1/
    ...
  # Chunk files stay in root (source-of-truth for re-compaction)
```

**Backwards compat:** When `num_shards` is absent in metadata, the loader reads from the root directory as before.

---

## 3. Search Pipeline Changes

### 3a. Loading (`rust/search/loader.rs`)

New method:
```rust
IndexLoader::load_sharded(index_path, devices: &[Device], dtype, use_mmap) -> Result<ShardedIndex>
```
- Loads centroids/bucket_weights once, `.to_device()` per shard
- Loads each shard's compacted files from `shard_i/`
- Computes local `offsets_compacted` per shard via cumsum

### 3b. Scorer (`rust/search/scorer.rs`)

New struct `ShardedWARPScorer`:
```rust
pub struct ShardedWARPScorer {
    sharded_index: ShardedIndex,
    shard_decompressors: Vec<CentroidDecompressor>,  // one per shard
    centroid_selector: CentroidSelector,              // shared (centroids replicated)
    merger: ResultMerger,                             // CPU merger for final merge
    config: SearchConfig,
}
```

**Search flow per query — recap of the single-shard pipeline:**
```
1. centroid_scores = Q @ centroids.T              → (q, centroid) rough scores
2. select top-nprobe centroids per token           → num_tokens * nprobe "cells"
3. decompress + fine-score each cell               → per-cell (pid, score) lists
4. merge_candidates_nprobe: max-reduce per (token, PID) across nprobe cells
5. merge_candidates_tokens: sum-reduce per PID across tokens (with MSE imputation)
6. top-k
```

**Sharded flow — exact behavioral equivalence:**
```
Coordinator (any single device, e.g. shard 0):
  1. centroid_scores = Q @ centroids.T             (centroids replicated, so any shard works)
  2. select_centroids → globally correct cell list  (num_tokens * nprobe cells)
     + mse_estimates per token

Per shard (parallel on each GPU):
  3. receive the full cell list; filter to cells whose centroid is in [start, end)
  4. translate global centroid IDs to local: local_id = global_id - centroid_start
  5. decompress + fine-score local cells
     → per-cell (pid, score) lists + cell indices (position in the global cell list)

Coordinator (CPU):
  6. gather per-cell results from all shards
  7. reassemble into the same cell-ordered structure as the single-shard case
     (each cell comes from exactly one shard — no overlap)
  8. run merge_candidates_nprobe (max-reduce across nprobe per token)  — unchanged
  9. run merge_candidates_tokens (sum-reduce across tokens with MSE)   — unchanged
  10. top-k
```

**Why this is exact:** Steps 1-2 are identical (same centroids, same selection). Each cell maps to exactly one centroid, which lives on exactly one shard, so the (q, d) scores in step 5 are identical to the single-shard case. Steps 8-10 operate on the same reassembled cell data. No approximation anywhere.

Steps 3-5 run in parallel across shards (one thread per shard, each dispatching to its GPU).

### 3c. Decompressor (`rust/search/decompressor.rs`)

One-line change gated on `shard_config`:
```rust
let lookup_ids = if let Some(sc) = &shard_config {
    centroid_ids - sc.centroid_start as i64
} else {
    centroid_ids.shallow_clone()
};
// use lookup_ids for offsets_compacted indexing
```

### 3d. Cell Gather + Merge (`rust/search/merger.rs`)

New function to reassemble shard results into the cell-ordered format expected by the existing merge:
```rust
pub fn gather_shard_cells(
    shard_cells: Vec<ShardCellOutput>,  // per-shard: (cell_indices, per-cell pid/score lists)
    num_cells: usize,                   // total = num_tokens * nprobe
) -> DecompressedCentroidsOutput
```
- Each shard returns its decompressed cells tagged with their position in the global cell list
- This function places them back into a single `DecompressedCentroidsOutput` ordered by cell index
- The existing `merge_candidate_scores` (CPU path) then runs unchanged on this output

No new merge algorithm needed — we reuse the existing merge exactly.

---

## 4. Index Creation / Compaction Changes

### 4a. Compaction (`rust/index/compact.rs`)

New function `compact_index_sharded(index_path, ..., shard_boundaries: &[usize])`:
- Same two-pass counting sort as current `compact_index`
- In pass 2, routes each centroid's data to the appropriate shard directory
- Writes per-shard `sizes/codes/residuals.compacted.npy`

### 4b. Creation (`rust/index/create.rs`)

Add optional `num_shards: usize` parameter to `create_index`. When > 1:
1. After encoding chunks, compute `shard_boundaries` using the balancing strategy (see section 6)
2. Call `compact_index_sharded` instead of `compact_index`
3. Write `shard_boundaries` and `num_shards` to `metadata.json`

### 4c. Offline re-sharding (new function)

```rust
pub fn shard_existing_index(index_path, num_shards, balance_strategy) -> Result<()>
```
Reads the existing monolithic compacted arrays, computes boundaries, slices into per-shard files. O(total_embeddings) copy, no re-encoding needed.

---

## 5. Incremental Update Changes (`rust/index/update.rs`)

**Add/update/compact** all operate on chunk files (root directory) then rebuild compacted arrays. The only change: the rebuild step calls `compact_index_sharded` or `merge_compacted_incremental_sharded` instead of the monolithic versions.

`merge_compacted_incremental_sharded` iterates over shards: for each shard, reads its old compacted files, merges the relevant centroid-range slice of the new `PartialCompacted`, writes updated shard files.

`append_centroids`: new centroids extend the last shard's range. `shard_boundaries` updated in metadata.

---

## 6. Shard Balancing

Default strategy: **equal embedding count** via greedy prefix-sum scan over `sizes.compacted`:

```rust
fn compute_balanced_boundaries(sizes: &[i64], num_shards: usize) -> Vec<usize>
```

This minimizes tail latency (the slowest shard dominates wall-clock time). After many incremental adds, shards may drift out of balance — `compact_standalone` can optionally recompute boundaries.

---

## 7. Python API Changes (`rust/lib.rs`)

New pyclass `ShardedSearcher` mirroring `LoadedSearcher`:
```python
searcher = ShardedSearcher(index_path, devices=["cuda:0", "cuda:1"], dtype="float16")
searcher.load()
results = searcher.search(torch_path, query_embeddings, search_config)
```

Updated `create()` pyfunction: add `num_shards: Option<usize>` (default `None`).

New `shard()` pyfunction for offline re-sharding of existing indices.

`LoadedSearcher` remains unchanged — 1-shard indices continue to use it as before.

---

## 8. Implementation Phases

| Phase | Scope | Key files |
|-------|-------|-----------|
| **1** | Data structures + on-disk format | `types.rs` |
| **2** | Sharded compaction | `compact.rs`, `create.rs` |
| **3** | Sharded loading | `loader.rs` |
| **4** | Sharded search (decompressor offset translation, shard filtering, global merge) | `decompressor.rs`, `scorer.rs`, `merger.rs` |
| **5** | Python API + `ShardedSearcher` | `lib.rs` |
| **6** | Sharded incremental updates | `update.rs`, `compact.rs` |
| **7** | Offline re-sharding utility | new function in `compact.rs` |

---

## 9. Verification

- **Exact equivalence test**: Create a sharded index (N=2+), run the same queries against it and a single-shard baseline. Results (PIDs and scores) must be **bitwise identical** (not approximate — this design guarantees exact equivalence).
- **N=1 equivalence test**: A 1-shard index loaded via `ShardedSearcher` must produce identical results to the same index loaded via `LoadedSearcher` (current path).
- **Memory test**: Verify each GPU only holds ~1/N of the compacted data (check `torch.cuda.memory_allocated` per device). Centroids + bucket_weights replicated on each.
- **Incremental test**: Add passages to a sharded index, verify search results include new passages, then compact and verify equivalence with single-shard.
