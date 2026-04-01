use anyhow::Result;
use rayon::{prelude::*, ThreadPool, ThreadPoolBuilder};
use std::sync::Arc;
use std::thread;
use tch::{Device, IndexOp, Kind, Tensor};

use crate::search::centroid_selector::CentroidSelector;
use crate::search::decompressor::CentroidDecompressor;
use crate::search::merger::{MergerConfig, ResultMerger};
use crate::search::merger::gather_shard_cells;
use crate::utils::types::{
    parse_device, parse_dtype, Query, ReadOnlyIndex, ReadOnlyTensor, SearchConfig, SearchResult,
    ShardCellOutput, ShardedIndex,
};

/// Main scorer struct that handles WARP scoring operations
/// This integrates the phase 1 components (CentroidSelector, CentroidDecompressor)
/// with the ranking pipeline
pub struct WARPScorer {
    /// Shared reference to the loaded index
    index: Arc<ReadOnlyIndex>,

    /// Centroid selector component from phase 1
    centroid_selector: CentroidSelector,

    /// Decompressor component from phase 1
    decompressor: CentroidDecompressor,

    /// Result merger for combining scores
    merger: ResultMerger,

    /// Configuration
    config: SearchConfig,

    /// Shared rayon thread pool
    thread_pool: Arc<ThreadPool>,

    /// Device to perform the scoring on
    device: tch::Device,

    /// Batch size for centroid matmul
    batch_size: i64,
}

impl WARPScorer {
    pub fn new(index: &Arc<ReadOnlyIndex>, config: SearchConfig) -> Result<Self> {
        let device = parse_device(&config.device)?;
        let batch_size = config.batch_size;
        let centroid_selector = CentroidSelector::new(
            &config,
            index.metadata.num_embeddings as usize,
            index.metadata.num_centroids,
        );
        let dtype = parse_dtype(&config.dtype)?;

        let num_threads = config
            .num_threads
            .unwrap_or_else(rayon::current_num_threads)
            .max(1);
        let thread_pool = Arc::new(ThreadPoolBuilder::new().num_threads(num_threads).build()?);

        let decompressor = CentroidDecompressor::new(
            index.metadata.nbits,
            index.metadata.dim,
            device,
            dtype,
            Arc::clone(&thread_pool),
        )?;

        let max_candidates = config.max_candidates.unwrap_or(256);
        let merger_config = MergerConfig {
            max_candidates: max_candidates,
            num_threads: config.num_threads.unwrap_or(1),
            device: device,
        };
        let merger = ResultMerger::new(merger_config);

        Ok(Self {
            index: Arc::clone(index),
            centroid_selector,
            decompressor,
            merger,
            config,
            thread_pool,
            device,
            batch_size,
        })
    }

    /// Process a single query
    #[inline]
    fn process_query(
        &self,
        query_idx: usize,
        query_embeddings: Tensor,
        centroid_scores: Tensor,
        query_mask: Tensor,
        k: usize,
    ) -> Result<SearchResult> {
        let selected = self.centroid_selector.select_centroids(
            &query_mask.to_device(self.device),
            &centroid_scores,
            &self.index.sizes_compacted,
            self.index.kdummy_centroid,
            k,
        )?;
        let decompressed = self.decompressor.decompress_centroids(
            &selected.centroid_ids.to_kind(Kind::Int64),
            &selected.scores,
            &self.index,
            &query_embeddings,
            self.config.nprobe as usize,
            None,
        )?;
        let (pids, scores) = self.merger.merge_candidate_scores(
            &decompressed.capacities,
            &decompressed.sizes,
            &decompressed.passage_ids,
            &decompressed.scores,
            &selected.mse_estimate,
            self.config.nprobe as usize,
            k,
        )?;

        Ok(SearchResult {
            passage_ids: pids,
            scores,
            query_id: query_idx + 1,
        })
    }

    /// Main ranking function that scores and ranks passages for a batch of queries
    pub fn rank(
        &self,
        query: &Query, // [batch, num_tokens, dim]
    ) -> Result<Vec<SearchResult>> {
        let _guard = tch::no_grad_guard();

        let k = self.config.k;
        let n_queries = query.embeddings.size()[0] as usize;
        // Need to wrap it to ensure it can be shared in the parallel path
        let masks: ReadOnlyTensor = ReadOnlyTensor(query.embeddings.ne(0).any_dim(2, false));

        if self.device == Device::Cpu {
            // cpu path: rayon works automatically with 1 thread
            let queries: Vec<ReadOnlyTensor> = (0..n_queries)
                .map(|b| ReadOnlyTensor(query.embeddings.select(0, b as i64)))
                .collect();

            let centroid_selector = self.centroid_selector.clone();
            let decompressor = self.decompressor.clone();
            let merger = self.merger.clone();
            let index = Arc::clone(&self.index);
            let nprobe = self.config.nprobe as usize;
            let device = self.device;

            self.thread_pool.install(move || {
                queries
                    .into_par_iter()
                    .enumerate()
                    .map(|(idx, query_embeddings)| {
                        let query_mask = masks.i(idx as i64);
                        let centroid_scores =
                            query_embeddings.matmul(&index.centroids.transpose(0, 1));

                        let selected = centroid_selector.select_centroids(
                            &query_mask.to_device(device),
                            &centroid_scores,
                            &index.sizes_compacted,
                            index.kdummy_centroid,
                            k,
                        )?;

                        let decompressed = decompressor.decompress_centroids(
                            &selected.centroid_ids.to_kind(Kind::Int64),
                            &selected.scores,
                            &index,
                            &query_embeddings,
                            nprobe,
                            None,
                        )?;

                        let (pids, scores) = merger.merge_candidate_scores(
                            &decompressed.capacities,
                            &decompressed.sizes,
                            &decompressed.passage_ids,
                            &decompressed.scores,
                            &selected.mse_estimate,
                            nprobe,
                            k,
                        )?;

                        Ok(SearchResult {
                            passage_ids: pids,
                            scores,
                            query_id: idx + 1,
                        })
                    })
                    .collect()
            })
        } else {
            // accelerator path (optimized for cuda)
            let mut results = Vec::with_capacity(n_queries);

            for c in (0..n_queries).step_by(self.batch_size as usize) {
                let batch_size = self.batch_size.min((n_queries - c) as i64);
                let batch_queries = query.embeddings.narrow(0, c as i64, batch_size);
                let batch_mask = masks.narrow(0, c as i64, batch_size);

                let centroid_scores = Tensor::einsum(
                    "btd,cd->btc",
                    &[&batch_queries, &self.index.centroids],
                    None::<&[i64]>,
                );

                for b in 0..batch_size {
                    // TODO this could be vectorized, with some effort
                    let result = self.process_query(
                        c + b as usize,
                        batch_queries.i(b),
                        centroid_scores.i(b),
                        batch_mask.i(b),
                        k,
                    )?;
                    results.push(result);
                }
            }
            Ok(results)
        }
    }
}

/// Sharded scorer that distributes decompression across multiple GPUs.
///
/// Centroid selection happens once on the coordinator (shard 0's device).
/// Each shard decompresses only the cells whose centroids fall in its range.
/// Results are gathered on CPU and merged using the existing merge pipeline.
pub struct ShardedWARPScorer {
    sharded_index: ShardedIndex,
    shard_decompressors: Vec<CentroidDecompressor>,
    centroid_selector: CentroidSelector,
    merger: ResultMerger,
    config: SearchConfig,
    coord_device: Device,
}

impl ShardedWARPScorer {
    pub fn new(sharded_index: ShardedIndex, config: SearchConfig) -> Result<Self> {
        let coord_device = parse_device(&config.device)?;
        let dtype = parse_dtype(&config.dtype)?;

        // Use shard 0 for centroid selection (centroids are replicated)
        let centroid_selector = CentroidSelector::new(
            &config,
            sharded_index.metadata.num_embeddings as usize,
            sharded_index.metadata.num_centroids,
        );

        let mut shard_decompressors = Vec::with_capacity(sharded_index.shards.len());
        for shard in &sharded_index.shards {
            let shard_device = if shard.shard_config.is_some() {
                shard.centroids.device()
            } else {
                coord_device
            };
            let pool = Arc::new(ThreadPoolBuilder::new().num_threads(1).build()?);
            shard_decompressors.push(CentroidDecompressor::new(
                sharded_index.metadata.nbits,
                sharded_index.metadata.dim,
                shard_device,
                dtype,
                pool,
            )?);
        }

        let max_candidates = config.max_candidates.unwrap_or(256);
        let merger = ResultMerger::new(MergerConfig {
            max_candidates,
            num_threads: config.num_threads.unwrap_or(1),
            device: coord_device,
        });

        Ok(Self {
            sharded_index,
            shard_decompressors,
            centroid_selector,
            merger,
            config,
            coord_device,
        })
    }

    /// Score a single query across all shards.
    fn process_query_sharded(
        &self,
        query_idx: usize,
        query_embeddings: &Tensor, // [num_tokens, dim]
        centroid_scores: &Tensor,  // [num_tokens, num_centroids]
        query_mask: &Tensor,
        k: usize,
    ) -> Result<SearchResult> {
        let nprobe = self.config.nprobe as usize;

        // Step 1-2: Select centroids (globally correct — centroids replicated)
        // For centroid selection, we need global sizes (sum across all shards).
        // Since each shard has local sizes, we build global sizes on the fly.
        let global_sizes = self.build_global_sizes()?;
        let kdummy = global_sizes.argmin(0, false).int64_value(&[]);

        let selected = self.centroid_selector.select_centroids(
            &query_mask.to_device(self.coord_device),
            &centroid_scores.to_device(self.coord_device),
            &global_sizes.to_device(self.coord_device),
            kdummy,
            k,
        )?;

        let centroid_ids: Vec<i64> = selected.centroid_ids.to_kind(Kind::Int64).try_into()?;
        let num_cells = centroid_ids.len();

        // Steps 3-5: Per-shard decompression (parallel via std::thread::scope)
        // Wrap query_embeddings in ReadOnlyTensor so it can be shared across threads.
        let query_ro = ReadOnlyTensor(query_embeddings.shallow_clone());
        let scores_vec: Vec<f32> = selected.scores.try_into()?;

        let num_shards = self.sharded_index.shards.len();
        let shard_outputs: Vec<Result<ShardCellOutput>> = thread::scope(|scope| {
            let handles: Vec<_> = (0..num_shards)
                .map(|s| {
                    let shard = &self.sharded_index.shards[s];
                    let sc = &self.sharded_index.shard_configs[s];
                    let decompressor = &self.shard_decompressors[s];
                    let centroid_ids = &centroid_ids;
                    let scores_vec = &scores_vec;
                    let query_ro = &query_ro;

                    scope.spawn(move || {
                        // Filter cells to this shard's centroid range
                        let mut local_cell_indices = Vec::new();
                        let mut local_centroid_ids = Vec::new();
                        let mut local_centroid_scores = Vec::new();
                        let mut local_token_indices = Vec::new();

                        for (cell_idx, &cid) in centroid_ids.iter().enumerate() {
                            let cid_usize = cid as usize;
                            if cid_usize >= sc.centroid_start && cid_usize < sc.centroid_end {
                                local_cell_indices.push(cell_idx);
                                local_centroid_ids.push(cid);
                                local_centroid_scores.push(scores_vec[cell_idx]);
                                // Preserve the global token assignment
                                local_token_indices.push(cell_idx / nprobe);
                            }
                        }

                        if local_centroid_ids.is_empty() {
                            return Ok(ShardCellOutput {
                                global_cell_indices: Vec::new(),
                                passage_ids: Vec::new(),
                                scores: Vec::new(),
                                capacities: Vec::new(),
                                sizes: Vec::new(),
                            });
                        }

                        let shard_device = shard.centroids.device();
                        let local_ids_tensor = Tensor::from_slice(&local_centroid_ids)
                            .to_device(shard_device);
                        let local_scores_tensor = Tensor::from_slice(&local_centroid_scores)
                            .to_device(shard_device);
                        let local_query = query_ro.to_device(shard_device);

                        let decompressed = decompressor.decompress_centroids(
                            &local_ids_tensor,
                            &local_scores_tensor,
                            shard,
                            &local_query,
                            nprobe,
                            Some(&local_token_indices),
                        )?;

                        // Convert to ShardCellOutput (move to CPU)
                        let pids: Vec<i64> = decompressed.passage_ids.to_device(Device::Cpu).try_into()?;
                        let scores: Vec<f32> = decompressed.scores.to_device(Device::Cpu).try_into()?;
                        let caps: Vec<i64> = decompressed.capacities.to_device(Device::Cpu).try_into()?;
                        let szs: Vec<i32> = decompressed.sizes.to_device(Device::Cpu).try_into()?;

                        Ok(ShardCellOutput {
                            global_cell_indices: local_cell_indices,
                            passage_ids: pids,
                            scores,
                            capacities: caps,
                            sizes: szs,
                        })
                    })
                })
                .collect();

            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        // Collect results, propagating errors
        let mut collected_outputs = Vec::with_capacity(num_shards);
        for result in shard_outputs {
            collected_outputs.push(result?);
        }

        // Steps 6-7: Gather shard results into unified cell structure on coordinator device
        let merge_device = self.coord_device;
        let gathered = gather_shard_cells(collected_outputs, num_cells, merge_device)?;

        // Steps 8-10: Merge (existing pipeline, unchanged)
        let mse_cpu = selected.mse_estimate.to_device(merge_device);
        let (pids, scores) = self.merger.merge_candidate_scores(
            &gathered.capacities,
            &gathered.sizes,
            &gathered.passage_ids,
            &gathered.scores,
            &mse_cpu,
            nprobe,
            k,
        )?;

        Ok(SearchResult {
            passage_ids: pids,
            scores,
            query_id: query_idx + 1,
        })
    }

    /// Build global sizes tensor by summing all shards' local sizes in centroid order.
    fn build_global_sizes(&self) -> Result<Tensor> {
        let num_centroids = self.sharded_index.metadata.num_centroids;
        let global = Tensor::zeros(&[num_centroids as i64], (Kind::Int64, Device::Cpu));
        for (s, shard) in self.sharded_index.shards.iter().enumerate() {
            let sc = &self.sharded_index.shard_configs[s];
            let local_sizes = shard.sizes_compacted.to_device(Device::Cpu).to_kind(Kind::Int64);
            let shard_len = local_sizes.size()[0];
            global
                .narrow(0, sc.centroid_start as i64, shard_len)
                .copy_(&local_sizes);
        }
        Ok(global)
    }

    /// Main ranking function for sharded search.
    pub fn rank(&self, query: &Query) -> Result<Vec<SearchResult>> {
        let _guard = tch::no_grad_guard();
        let k = self.config.k;
        let n_queries = query.embeddings.size()[0] as usize;

        let masks = query.embeddings.ne(0).any_dim(2, false);

        let mut results = Vec::with_capacity(n_queries);

        for q in 0..n_queries {
            let q_emb = query.embeddings.select(0, q as i64);
            let q_mask = masks.select(0, q as i64);

            // Compute centroid scores on coordinator device
            let q_emb_coord = q_emb.to_device(self.coord_device);
            let centroid_scores = q_emb_coord
                .matmul(&self.sharded_index.shards[0].centroids.transpose(0, 1));

            let result = self.process_query_sharded(
                q,
                &q_emb_coord,
                &centroid_scores,
                &q_mask,
                k,
            )?;
            results.push(result);
        }

        Ok(results)
    }
}
