// Key types needed:
// - Configuration structures for index and search
// - ID types for passages, centroids, embeddings
// - Query and result representations
// - Index components structure
use memmap2::Mmap;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tch::{Device, Kind, Tensor};

/// Represents a passage/document ID in the index
pub type PassageId = i64;

/// Represents a centroid ID in the clustering
pub type CentroidId = i64;

/// Represents an embedding ID in the index
pub type EmbeddingId = i64;

/// Query ID type
pub type QueryId = i64;

/// Score type for ranking
pub type Score = f32;

/// Configuration for the WARP index
#[derive(Debug, Clone)]
pub struct IndexConfig {
    /// Path to the index directory
    pub index_path: PathBuf,

    /// Device to use (e.g. "cpu", "cuda:0")
    pub device: Device,

    /// Number of bits for residual compression (2 or 4)
    pub nbits: u8,

    /// Embedding dimension
    pub embedding_dim: u32,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            index_path: PathBuf::from("./index"),
            device: Device::Cpu,
            nbits: 4,
            embedding_dim: 128,
        }
    }
}

/// Search configuration parameters
#[pyclass]
#[derive(Debug, Clone)]
pub struct SearchConfig {
    /// Number of top results to return
    #[pyo3(get, set)]
    pub k: usize,

    /// Device to use for the search
    #[pyo3(get, set)]
    pub device: String,

    /// Dtype to use for the search
    #[pyo3(get, set)]
    pub dtype: String,

    /// Number of centroids to probe during search
    #[pyo3(get, set)]
    pub nprobe: u32,

    /// Optional adaptive threshold (t')
    #[pyo3(get, set)]
    pub t_prime: Option<usize>,

    /// Maximum number of centroids to inspect per token
    #[pyo3(get, set)]
    pub bound: usize,

    /// The batch size for centroid matmul
    #[pyo3(get, set)]
    pub batch_size: i64,

    /// Optional number of threads for parallel search
    #[pyo3(get, set)]
    pub num_threads: Option<usize>,

    /// Threshold for centroid scores
    #[pyo3(get, set)]
    pub centroid_score_threshold: Option<f32>,

    /// Maximum codes per centroid
    #[pyo3(get, set)]
    pub max_codes_per_centroid: Option<u32>,

    /// The number of candidates to consider before the sorting
    #[pyo3(get, set)]
    pub max_candidates: Option<usize>,
}

#[pymethods]
impl SearchConfig {
    /// Creates a new SearchConfig instance from Python
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        k,
        device,
        dtype=None,
        nprobe=None,
        t_prime=None,
        bound=None,
        batch_size=None,
        num_threads=None,
        centroid_score_threshold=None,
        max_codes_per_centroid=None,
        max_candidates=None,
    ))]
    fn new(
        k: usize,
        device: String,
        dtype: Option<String>,
        nprobe: Option<u32>,
        t_prime: Option<usize>,
        bound: Option<usize>,
        batch_size: Option<i64>,
        num_threads: Option<usize>,
        centroid_score_threshold: Option<f32>,
        max_codes_per_centroid: Option<u32>,
        max_candidates: Option<usize>,
    ) -> Self {
        Self {
            k,
            device: device,
            dtype: dtype.unwrap_or("float32".to_string()),
            nprobe: nprobe.unwrap_or(4),
            t_prime,
            bound: bound.unwrap_or(128),
            batch_size: batch_size.unwrap_or(8192i64),
            num_threads,
            centroid_score_threshold,
            max_codes_per_centroid,
            max_candidates,
        }
    }
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            k: 100,
            device: "cpu".to_string(),
            dtype: "float32".to_string(),
            nprobe: 4,
            t_prime: None,
            bound: 128,
            batch_size: 8192i64,
            num_threads: Some(1usize),
            centroid_score_threshold: None,
            max_codes_per_centroid: None,
            max_candidates: None,
        }
    }
}

/// Represents a ranked search result
#[pyclass]
#[derive(Serialize, Debug, Clone)]
pub struct SearchResult {
    #[pyo3(get)]
    pub passage_ids: Vec<PassageId>,
    #[pyo3(get)]
    pub scores: Vec<Score>,
    #[pyo3(get)]
    pub query_id: usize,
}

/// Configuration for a single shard within a sharded index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardConfig {
    /// This shard's index (0..num_shards-1)
    pub shard_id: usize,
    /// Total number of shards
    pub num_shards: usize,
    /// First centroid ID owned by this shard (inclusive)
    pub centroid_start: usize,
    /// Last centroid ID owned by this shard (exclusive)
    pub centroid_end: usize,
}

/// A sharded index: multiple ReadOnlyIndex instances (one per shard), each
/// holding replicated centroids/bucket_weights but only its slice of the
/// compacted arrays.
#[derive(Clone)]
pub struct ShardedIndex {
    pub shards: Vec<Arc<ReadOnlyIndex>>,
    pub shard_configs: Vec<ShardConfig>,
    pub metadata: IndexMetadata,
}

/// Represents the plan outlined to create an index
pub struct IndexPlan {
    pub n_docs: usize,
    pub num_chunks: usize,
    pub avg_doc_len: f64,
    pub est_total_embs: i64,
    pub nbits: u8,
}

/// Canonical on-disk + in-memory representation of `metadata.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMetadata {
    pub num_chunks: usize,
    pub nbits: u8,
    pub num_partitions: i64,
    pub num_embeddings: i64,
    pub avg_doclen: f64,
    pub num_passages: usize,
    /// Watermark for the next passage ID to assign.
    pub next_passage_id: i64,
    pub num_centroids: usize,
    pub dim: usize,
    pub created_at: String,
    /// Number of shards (absent or 1 for legacy single-shard indices).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_shards: Option<usize>,
    /// Centroid boundaries per shard: [0, b1, b2, ..., num_centroids].
    /// Length is num_shards + 1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shard_boundaries: Option<Vec<usize>>,
}

impl IndexMetadata {
    /// Load metadata from `metadata.json` in the given index directory.
    pub fn load(index_path: &std::path::Path) -> anyhow::Result<Self> {
        let path = index_path.join("metadata.json");
        let file = std::fs::File::open(&path)
            .map_err(|e| anyhow::anyhow!("Failed to open {}: {}", path.display(), e))?;
        let meta: IndexMetadata = serde_json::from_reader(std::io::BufReader::new(file))
            .map_err(|e| anyhow::anyhow!("Failed to parse {}: {}", path.display(), e))?;
        Ok(meta)
    }

    /// Persist metadata to `metadata.json` atomically (write tmp, rename).
    pub fn save(&self, index_path: &std::path::Path) -> anyhow::Result<()> {
        let path = index_path.join("metadata.json");
        let tmp_path = index_path.join("metadata.json.tmp");
        let file = std::fs::File::create(&tmp_path)
            .map_err(|e| anyhow::anyhow!("Failed to create {}: {}", tmp_path.display(), e))?;
        serde_json::to_writer_pretty(std::io::BufWriter::new(file), self)?;
        std::fs::rename(&tmp_path, &path)
            .map_err(|e| anyhow::anyhow!("Failed to rename {} -> {}: {}", tmp_path.display(), path.display(), e))?;
        Ok(())
    }
}

/// Components of a loaded index
pub struct LoadedIndex {
    /// Centroids tensor [num_centroids, dim]
    pub centroids: Tensor,

    /// Bucket weights for scoring
    pub bucket_weights: Tensor,

    /// Compacted sizes per centroid
    pub sizes_compacted: Tensor,

    /// Per-embedding passage IDs in compacted (centroid-sorted) layout
    pub pids_compacted: Tensor,

    /// Compacted residuals (compressed)
    pub residuals_compacted: Tensor,

    /// Offsets for each centroid in the compacted arrays
    pub offsets_compacted: Tensor,

    /// Index of the dummy centroid (smallest)
    pub kdummy_centroid: CentroidId,

    /// Metadata about the index
    pub metadata: IndexMetadata,

    /// Mmap handles that must outlive the tensors they back.
    /// Arc-wrapped for shared ownership across shallow clones.
    pub _mmap_handles: Arc<Vec<Mmap>>,

    /// Shard configuration, if this index is one shard of a sharded index.
    /// Used by the decompressor to translate global centroid IDs to local offsets.
    pub shard_config: Option<ShardConfig>,
}

impl Clone for LoadedIndex {
    fn clone(&self) -> Self {
        Self {
            centroids: self.centroids.shallow_clone(),
            bucket_weights: self.bucket_weights.shallow_clone(),
            sizes_compacted: self.sizes_compacted.shallow_clone(),
            pids_compacted: self.pids_compacted.shallow_clone(),
            residuals_compacted: self.residuals_compacted.shallow_clone(),
            offsets_compacted: self.offsets_compacted.shallow_clone(),
            kdummy_centroid: self.kdummy_centroid,
            metadata: self.metadata.clone(),
            _mmap_handles: Arc::clone(&self._mmap_handles),
            shard_config: self.shard_config.clone(),
        }
    }
}

/// Read-only wrapper that marks the loaded index as safe to share across threads.
/// The tensors are never mutated after load, so we can treat them as Sync.
pub struct ReadOnlyIndex(pub LoadedIndex);

impl std::ops::Deref for ReadOnlyIndex {
    type Target = LoadedIndex;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

unsafe impl Sync for ReadOnlyIndex {}

impl Clone for ReadOnlyIndex {
    fn clone(&self) -> Self {
        ReadOnlyIndex(self.0.clone())
    }
}

/// Query representation for search
pub struct Query {
    pub embeddings: Tensor, // Always [batch, num_tokens, dim]
}

/// Batch of queries for efficient processing
pub struct QueryBatch {
    pub queries: Vec<Query>,
    pub max_tokens: usize,
}

/// Read-only tensor wrapper to opt into Sync when we guarantee no mutation.
pub struct ReadOnlyTensor(pub Tensor);

impl std::ops::Deref for ReadOnlyTensor {
    type Target = Tensor;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

unsafe impl Sync for ReadOnlyTensor {}

impl Clone for ReadOnlyTensor {
    fn clone(&self) -> Self {
        ReadOnlyTensor(self.0.shallow_clone())
    }
}

/// Selected centroids for a query token
#[derive(Debug)]
pub struct SelectedCentroids {
    pub centroid_ids: Tensor,
    pub scores: Tensor,
    pub mse_estimate: Tensor,
}

/// Decompressed centroid data
pub struct DecompressedCentroid {
    pub centroid_id: CentroidId,
    pub passage_ids: Vec<PassageId>,
    pub scores: Vec<Score>,
}

pub struct DecompressedCentroidsOutput {
    pub capacities: Tensor, // Total capacity of each centroid (ends - begins)
    pub sizes: Tensor,      // Actual sizes after deduplication
    pub passage_ids: Tensor,
    pub scores: Tensor,
    pub offsets: Tensor,
}

/// Candidate for final ranking
#[derive(Debug, Clone)]
pub struct RankingCandidate {
    pub passage_id: PassageId,
    pub score: Score,
    pub centroid_id: CentroidId,
}

/// T-prime policy for adaptive early termination
#[derive(Debug, Clone)]
pub enum TPrimePolicy {
    Fixed(usize),
    Max,
}

impl TPrimePolicy {
    pub fn value(&self, k: usize) -> usize {
        match self {
            TPrimePolicy::Fixed(value) => *value,
            TPrimePolicy::Max => {
                if k > 100 {
                    100_000
                } else {
                    50_000
                }
            },
        }
    }
}

/// Stats from index compaction
pub struct CompactStats {
    pub total_embeddings: i64,
    pub num_active_passages: usize,
}

/// Result from adding documents to an index
pub struct AddResult {
    pub new_passage_ids: Vec<i64>,
    /// Per-embedding residual norms (for centroid expansion outlier detection).
    pub residual_norms: Vec<f32>,
    /// Embedding dimension.
    pub embedding_dim: usize,
}

/// Per-shard decompression output, tagged with global cell indices so the
/// coordinator can reassemble them into a single DecompressedCentroidsOutput.
pub struct ShardCellOutput {
    /// Positions of this shard's cells in the global cell list (0..num_tokens*nprobe).
    pub global_cell_indices: Vec<usize>,
    /// Per-cell passage IDs (concatenated).
    pub passage_ids: Vec<PassageId>,
    /// Per-cell scores (concatenated, same length as passage_ids).
    pub scores: Vec<Score>,
    /// Per-cell capacities (total range, before dedup).
    pub capacities: Vec<i64>,
    /// Per-cell sizes (after dedup).
    pub sizes: Vec<i32>,
}

/// Compute shard boundaries that balance the total number of embeddings across
/// shards. Returns a Vec of length `num_shards + 1` where `boundaries[i]` is
/// the first centroid ID of shard `i` and `boundaries[num_shards] == num_centroids`.
pub fn compute_balanced_boundaries(sizes: &[i64], num_shards: usize) -> Vec<usize> {
    assert!(num_shards > 0, "num_shards must be > 0");
    let num_centroids = sizes.len();
    if num_shards >= num_centroids {
        // More shards than centroids: one centroid per shard (some empty).
        let mut boundaries: Vec<usize> = (0..=num_centroids).collect();
        // Pad with empty shards at the end.
        while boundaries.len() < num_shards + 1 {
            boundaries.push(num_centroids);
        }
        return boundaries;
    }

    let total: i64 = sizes.iter().sum();
    let target_per_shard = total as f64 / num_shards as f64;

    let mut boundaries = Vec::with_capacity(num_shards + 1);
    boundaries.push(0);

    let mut running: i64 = 0;
    for (i, &s) in sizes.iter().enumerate() {
        running += s;
        // Place a boundary when we've accumulated enough, but don't exceed num_shards.
        if running as f64 >= target_per_shard && boundaries.len() < num_shards {
            boundaries.push(i + 1);
            running = 0;
        }
    }
    boundaries.push(num_centroids);

    // In degenerate cases we may not have placed enough boundaries; fill in.
    while boundaries.len() < num_shards + 1 {
        let last = *boundaries.last().unwrap();
        boundaries.push(last);
    }

    boundaries
}

/// Parses a string identifier into a `tch::Device`.
///
/// Supports simple device strings like "cpu", "cuda", and indexed CUDA devices
/// such as "cuda:0".
pub fn parse_device(device: &str) -> anyhow::Result<Device> {
    match device.to_lowercase().as_str() {
        "cpu" => Ok(Device::Cpu),
        "mps" => Ok(Device::Mps),
        "cuda" => Ok(Device::Cuda(0)), // Default to the first CUDA device.
        s if s.starts_with("cuda:") => {
            let parts: Vec<&str> = s.split(':').collect();
            if parts.len() == 2 {
                parts[1]
                    .parse::<usize>()
                    .map(Device::Cuda)
                    .map_err(|_| anyhow::anyhow!("Invalid CUDA device index: '{}'", parts[1]))
            } else {
                Err(anyhow::anyhow!(
                    "Invalid CUDA device format. Expected 'cuda:N'."
                ))
            }
        },
        _ => Err(anyhow::anyhow!("Unsupported device string: '{}'", device)),
    }
}

/// Parses a string identifier into a `tch::Kind`.
///
/// Supports simple strings like "float32", "float16"
pub fn parse_dtype(dtype: &str) -> anyhow::Result<Kind> {
    match dtype.to_lowercase().as_str() {
        "float32" => Ok(Kind::Float),
        "float16" => Ok(Kind::Half),
        "float64" => Ok(Kind::Double),
        "bfloat16" => Ok(Kind::BFloat16),
        _ => Err(anyhow::anyhow!("Unsupported dtype string: '{}', should be 'float32', 'float16', 'float64', or 'bfloat16'", dtype)),
    }
}
