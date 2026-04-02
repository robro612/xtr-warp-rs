use anyhow::Result;
use serde_json::json;
use std::env;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use tch::{Device, Kind, Tensor};

/// Data from a single chunk, used for coalescing during add.
pub struct ChunkData {
    pub codes: Tensor,
    pub residuals: Tensor,
    pub doclens: Vec<i64>,
    pub pids: Vec<i64>,
}

/// Read the lightweight chunk files (codes, residuals, doclens, passage IDs).
pub fn read_chunk_data(index_path: &Path, chunk_idx: usize) -> Result<ChunkData> {
    let codes = Tensor::read_npy(index_path.join(format!("{}.codes.npy", chunk_idx)))?
        .to_device(tch::Device::Cpu);
    let residuals = Tensor::read_npy(index_path.join(format!("{}.residuals.npy", chunk_idx)))?
        .to_device(tch::Device::Cpu);
    let doclens: Vec<i64> = Tensor::read_npy(index_path.join(format!("doclens.{}.npy", chunk_idx)))?
        .to_device(tch::Device::Cpu)
        .to_kind(Kind::Int64)
        .try_into()?;
    let pids: Vec<i64> = Tensor::read_npy(index_path.join(format!("{}.passage_ids.npy", chunk_idx)))?
        .to_device(tch::Device::Cpu)
        .to_kind(Kind::Int64)
        .try_into()?;
    Ok(ChunkData { codes, residuals, doclens, pids })
}

use crate::index::source::EmbeddingSource;
use crate::utils::residual_codec::ResidualCodec;
use crate::utils::types::IndexPlan;

pub const CHUNK_SIZE: usize = 25_000;
pub const EMB_BATCH_SIZE: i64 = 1 << 18;
pub const CODE_BATCH_SIZE: i64 = 1 << 20;

const BIT_WEIGHTS: [i64; 8] = [128, 64, 32, 16, 8, 4, 2, 1];

pub struct EncodeResult {
    pub chunk_stats: Vec<ChunkStats>,
    pub total_embeddings: i64,
    pub global_centroid_counts: Tensor,
    /// Per-embedding L2 residual norms (only populated when `collect_norms` is true).
    pub residual_norms: Option<Vec<f32>>,
}

pub struct ChunkStats {
    pub embedding_offset: usize,
    pub num_embeddings: usize,
}

pub fn encode_chunks(
    plan: &IndexPlan,
    source: &mut dyn EmbeddingSource,
    centroids: &Tensor,
    codec: &ResidualCodec,
    index_path: &Path,
    device: Device,
    embedding_dim: u32,
    passage_ids: Option<&[i64]>,
    start_chunk_idx: usize,
    num_shards: Option<usize>,
) -> Result<EncodeResult> {
    encode_chunks_inner(
        plan,
        source,
        centroids,
        codec,
        index_path,
        device,
        embedding_dim,
        passage_ids,
        start_chunk_idx,
        num_shards,
        false,
    )
}

/// Like `encode_chunks` but also returns per-embedding residual norms.
pub fn encode_chunks_with_norms(
    plan: &IndexPlan,
    source: &mut dyn EmbeddingSource,
    centroids: &Tensor,
    codec: &ResidualCodec,
    index_path: &Path,
    device: Device,
    embedding_dim: u32,
    passage_ids: Option<&[i64]>,
    start_chunk_idx: usize,
    num_shards: Option<usize>,
) -> Result<EncodeResult> {
    encode_chunks_inner(
        plan,
        source,
        centroids,
        codec,
        index_path,
        device,
        embedding_dim,
        passage_ids,
        start_chunk_idx,
        num_shards,
        true,
    )
}

fn effective_emb_batch_size(num_shards: Option<usize>) -> i64 {
    if let Ok(raw) = env::var("XTR_WARP_ENCODE_EMB_BATCH_SIZE") {
        if let Ok(parsed) = raw.parse::<i64>() {
            if parsed > 0 {
                return parsed;
            }
        }
    }

    let shard_factor = num_shards.unwrap_or(1).max(1) as i64;
    let scaled = EMB_BATCH_SIZE / shard_factor;
    // Keep a sane floor so extremely high shard counts don't become pathological.
    scaled.max(1 << 15)
}

fn parse_env_i64(name: &str) -> Option<i64> {
    env::var(name).ok()?.parse::<i64>().ok().filter(|v| *v > 0)
}

fn encode_profile_enabled() -> bool {
    matches!(
        env::var("XTR_WARP_PROFILE_ENCODE").ok().as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES")
    )
}

fn encode_local_profile_enabled() -> bool {
    matches!(
        env::var("XTR_WARP_PROFILE_ENCODE_LOCAL")
            .ok()
            .as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES")
    )
}

fn kind_size_bytes(kind: Kind) -> usize {
    match kind {
        Kind::Bool => 1,
        Kind::Uint8 | Kind::Int8 => 1,
        Kind::Int16 | Kind::Half | Kind::BFloat16 => 2,
        Kind::Int | Kind::Float => 4,
        Kind::Int64 | Kind::Double => 8,
        _ => 4,
    }
}

fn tensor_nbytes(t: &Tensor) -> usize {
    (t.numel() as usize) * kind_size_bytes(t.kind())
}

fn mb(bytes: usize) -> f64 {
    (bytes as f64) / (1024.0 * 1024.0)
}

#[derive(Default)]
struct EncodeLocalStats {
    max_emb_batch_bytes: usize,
    max_score_est_bytes: usize,
    max_code_batch_bytes: usize,
    max_recon_bytes: usize,
    max_residual_float_bytes: usize,
    max_residual_bucketized_bytes: usize,
    max_residual_bits_bytes: usize,
    max_residual_packed_bytes: usize,
    max_working_set_est_bytes: usize,
}

impl EncodeLocalStats {
    fn update_batch(
        &mut self,
        emb_batch_bytes: usize,
        score_est_bytes: usize,
        code_batch_bytes: usize,
        recon_bytes: usize,
        residual_float_bytes: usize,
        residual_bucketized_bytes: usize,
        residual_bits_bytes: usize,
        residual_packed_bytes: usize,
    ) {
        self.max_emb_batch_bytes = self.max_emb_batch_bytes.max(emb_batch_bytes);
        self.max_score_est_bytes = self.max_score_est_bytes.max(score_est_bytes);
        self.max_code_batch_bytes = self.max_code_batch_bytes.max(code_batch_bytes);
        self.max_recon_bytes = self.max_recon_bytes.max(recon_bytes);
        self.max_residual_float_bytes =
            self.max_residual_float_bytes.max(residual_float_bytes);
        self.max_residual_bucketized_bytes = self
            .max_residual_bucketized_bytes
            .max(residual_bucketized_bytes);
        self.max_residual_bits_bytes = self.max_residual_bits_bytes.max(residual_bits_bytes);
        self.max_residual_packed_bytes = self.max_residual_packed_bytes.max(residual_packed_bytes);

        // Rough upper bound of simultaneously-live tensors in the hot loop.
        let working_est = emb_batch_bytes
            + score_est_bytes
            + code_batch_bytes
            + recon_bytes
            + residual_float_bytes
            + residual_bucketized_bytes
            + residual_bits_bytes
            + residual_packed_bytes;
        self.max_working_set_est_bytes = self.max_working_set_est_bytes.max(working_est);
    }
}

fn effective_code_batch_size(num_shards: Option<usize>) -> i64 {
    if let Some(v) = parse_env_i64("XTR_WARP_ENCODE_CODE_BATCH_SIZE") {
        return v;
    }
    let shard_factor = num_shards.unwrap_or(1).max(1) as i64;
    let scaled = CODE_BATCH_SIZE / shard_factor;
    scaled.max(1 << 14)
}

fn effective_score_batch_size(num_centroids: i64, num_shards: Option<usize>) -> i64 {
    if let Some(v) = parse_env_i64("XTR_WARP_ENCODE_SCORE_BATCH_SIZE") {
        return v;
    }

    let shard_factor = num_shards.unwrap_or(1).max(1) as i64;
    let max_score_elems = parse_env_i64("XTR_WARP_MAX_SCORE_ELEMS").unwrap_or(1 << 29);
    let scaled_elems = (max_score_elems / shard_factor).max(1);
    (scaled_elems / num_centroids.max(1)).max(1)
}

#[allow(clippy::too_many_arguments)]
fn encode_embedding_batch(
    emb_batch: &Tensor,
    codec: &ResidualCodec,
    plan: &IndexPlan,
    embedding_dim: u32,
    device: Device,
    num_centroids: usize,
    collect_norms: bool,
    all_norms: &mut Vec<f32>,
    chk_codes_list: &mut Vec<Tensor>,
    chk_res_list: &mut Vec<Tensor>,
    global_counts: &mut Tensor,
    code_batch_size: i64,
    score_batch_size: i64,
    local_stats: Option<&mut EncodeLocalStats>,
) -> Result<()> {
    let code_batch = compress_into_codes_with_batch(emb_batch, &codec.centroids, score_batch_size);
    let chunk_counts = code_batch.bincount::<Tensor>(None, num_centroids as i64);
    *global_counts = &*global_counts + &chunk_counts;
    chk_codes_list.push(code_batch.to_device(Device::Cpu));

    let mut recon_centroids_batches: Vec<Tensor> = Vec::new();
    for sub_code_batch in code_batch.split(code_batch_size, 0) {
        recon_centroids_batches.push(codec.centroids.index_select(0, &sub_code_batch));
    }
    let recon_centroids = Tensor::cat(&recon_centroids_batches, 0);

    let mut res_batch = emb_batch - &recon_centroids;
    let residual_float_bytes = tensor_nbytes(&res_batch);
    if collect_norms {
        let norms = res_batch
            .to_kind(Kind::Float)
            .norm_scalaropt_dim(2, &[1], false)
            .to_device(Device::Cpu);
        let norms_vec: Vec<f32> = norms.try_into()?;
        all_norms.extend(norms_vec);
    }

    let bucket_cutoffs = codec.bucket_cutoffs.as_ref().unwrap().contiguous();
    res_batch = Tensor::bucketize(&res_batch, &bucket_cutoffs, true, false);
    let residual_bucketized_bytes = tensor_nbytes(&res_batch);

    let mut res_shape = res_batch.size();
    res_shape.push(plan.nbits as i64);
    res_batch = res_batch.unsqueeze(-1).expand(&res_shape, false);
    res_batch = res_batch.bitwise_right_shift(&codec.arange_bits);
    let ones = Tensor::ones_like(&res_batch).to_device(device);
    res_batch = res_batch.bitwise_and_tensor(&ones);
    let residual_bits_bytes = tensor_nbytes(&res_batch);

    let res_flat = res_batch.flatten(0, -1);
    let res_packed = packbits(&res_flat);
    let residual_packed_bytes = tensor_nbytes(&res_packed);
    let shape = [
        res_batch.size()[0],
        (embedding_dim as i64) / 8 * (plan.nbits as i64),
    ];
    chk_res_list.push(res_packed.reshape(shape).to_device(Device::Cpu));

    if let Some(stats) = local_stats {
        let emb_batch_bytes = tensor_nbytes(emb_batch);
        let score_est_bytes = (codec.centroids.size()[0] as usize)
            * (emb_batch.size()[0] as usize)
            * kind_size_bytes(Kind::Half);
        let code_batch_bytes = tensor_nbytes(&code_batch);
        let recon_bytes = tensor_nbytes(&recon_centroids);
        stats.update_batch(
            emb_batch_bytes,
            score_est_bytes,
            code_batch_bytes,
            recon_bytes,
            residual_float_bytes,
            residual_bucketized_bytes,
            residual_bits_bytes,
            residual_packed_bytes,
        );
    }
    Ok(())
}

fn encode_chunks_inner(
    plan: &IndexPlan,
    source: &mut dyn EmbeddingSource,
    centroids: &Tensor,
    codec: &ResidualCodec,
    index_path: &Path,
    device: Device,
    embedding_dim: u32,
    passage_ids: Option<&[i64]>,
    start_chunk_idx: usize,
    num_shards: Option<usize>,
    collect_norms: bool,
) -> Result<EncodeResult> {
    if let Some(pids) = passage_ids {
        anyhow::ensure!(
            pids.len() == source.num_docs(),
            "passage_ids length ({}) must match source num_docs ({})",
            pids.len(),
            source.num_docs()
        );
    }

    let num_centroids = centroids.size()[0] as usize;
    let mut chunk_stats = Vec::with_capacity(plan.num_chunks);
    let mut current_emb_offset: usize = 0;
    let mut total_embeddings: i64 = 0;
    let mut global_counts = Tensor::zeros(&[num_centroids as i64], (Kind::Int64, device));
    let mut passage_offset: usize = 0;
    let mut all_norms: Vec<f32> = Vec::new();
    let emb_batch_size = effective_emb_batch_size(num_shards);
    let code_batch_size = effective_code_batch_size(num_shards);
    let score_batch_size = effective_score_batch_size(num_centroids as i64, num_shards);
    let mut local_stats = EncodeLocalStats::default();
    if encode_profile_enabled() {
        eprintln!(
            "[encode-profile] num_shards={:?} num_centroids={} emb_batch_size={} score_batch_size={} code_batch_size={}",
            num_shards,
            num_centroids,
            emb_batch_size,
            score_batch_size,
            code_batch_size
        );
    }

    let chunk_iter = source.chunk_iter(CHUNK_SIZE)?;
    for (local_chk_idx, chunk) in chunk_iter.enumerate() {
        let chk_idx = start_chunk_idx + local_chk_idx;
        let chunk = chunk?;
        let chk_doclens = chunk.doclens;
        let chk_embs_vec = chunk.embeddings;

        let mut chk_codes_list: Vec<Tensor> = Vec::new();
        let mut chk_res_list: Vec<Tensor> = Vec::new();
        let mut chunk_num_embeddings: usize = 0;

        // Stream docs into bounded microbatches to avoid materializing a full chunk on GPU.
        let mut micro_docs: Vec<Tensor> = Vec::new();
        let mut micro_rows: i64 = 0;
        for doc in chk_embs_vec {
            micro_rows += doc.size()[0];
            micro_docs.push(doc);

            if micro_rows < emb_batch_size {
                continue;
            }

            let refs: Vec<&Tensor> = micro_docs.iter().collect();
            let micro = Tensor::cat(&refs, 0).to_kind(Kind::Half).to_device(device);
            total_embeddings += micro.size()[0];
            chunk_num_embeddings += micro.size()[0] as usize;
            for emb_batch in micro.split(emb_batch_size, 0) {
                encode_embedding_batch(
                    &emb_batch,
                    codec,
                    plan,
                    embedding_dim,
                    device,
                    num_centroids,
                    collect_norms,
                    &mut all_norms,
                    &mut chk_codes_list,
                    &mut chk_res_list,
                    &mut global_counts,
                    code_batch_size,
                    score_batch_size,
                    if encode_local_profile_enabled() {
                        Some(&mut local_stats)
                    } else {
                        None
                    },
                )?;
            }
            micro_docs.clear();
            micro_rows = 0;
        }

        if !micro_docs.is_empty() {
            let refs: Vec<&Tensor> = micro_docs.iter().collect();
            let micro = Tensor::cat(&refs, 0).to_kind(Kind::Half).to_device(device);
            total_embeddings += micro.size()[0];
            chunk_num_embeddings += micro.size()[0] as usize;
            for emb_batch in micro.split(emb_batch_size, 0) {
                encode_embedding_batch(
                    &emb_batch,
                    codec,
                    plan,
                    embedding_dim,
                    device,
                    num_centroids,
                    collect_norms,
                    &mut all_norms,
                    &mut chk_codes_list,
                    &mut chk_res_list,
                    &mut global_counts,
                    code_batch_size,
                    score_batch_size,
                    if encode_local_profile_enabled() {
                        Some(&mut local_stats)
                    } else {
                        None
                    },
                )?;
            }
        }

        let chk_codes = Tensor::cat(&chk_codes_list, 0);
        let chk_residuals = Tensor::cat(&chk_res_list, 0);

        let chk_codes_fpath = index_path.join(&format!("{}.codes.npy", chk_idx));
        chk_codes.write_npy(&chk_codes_fpath)?;

        let chk_res_fpath = index_path.join(&format!("{}.residuals.npy", chk_idx));
        chk_residuals.write_npy(&chk_res_fpath)?;

        let chk_doclens_fpath = index_path.join(format!("doclens.{}.npy", chk_idx));
        Tensor::from_slice(&chk_doclens).write_npy(chk_doclens_fpath)?;

        // Write explicit passage IDs for this chunk
        let chunk_pids: Vec<i64> = if let Some(pids) = passage_ids {
            pids[passage_offset..passage_offset + chk_doclens.len()].to_vec()
        } else {
            (passage_offset as i64..(passage_offset + chk_doclens.len()) as i64).collect()
        };
        let chunk_pids_fpath = index_path.join(format!("{}.passage_ids.npy", chk_idx));
        Tensor::from_slice(&chunk_pids).write_npy(&chunk_pids_fpath)?;

        let chk_meta = json!({
            "passage_offset": passage_offset,
            "num_passages": chk_doclens.len(),
            "num_embeddings": chunk_num_embeddings,
            "embedding_offset": current_emb_offset,
        });
        let chk_meta_fpath = index_path.join(format!("{}.metadata.json", chk_idx));
        let meta_f_w = File::create(chk_meta_fpath)?;
        let buf_writer_meta = BufWriter::new(meta_f_w);
        serde_json::to_writer(buf_writer_meta, &chk_meta)?;

        chunk_stats.push(ChunkStats {
            embedding_offset: current_emb_offset,
            num_embeddings: chunk_num_embeddings,
        });
        current_emb_offset += chunk_num_embeddings;
        passage_offset += chk_doclens.len();
    }

    let result = EncodeResult {
        chunk_stats,
        total_embeddings,
        global_centroid_counts: global_counts,
        residual_norms: if collect_norms { Some(all_norms) } else { None },
    };
    if encode_local_profile_enabled() {
        eprintln!(
            "[encode-local-profile] num_shards={:?} max_emb_batch_mb={:.1} max_score_est_mb={:.1} max_code_batch_mb={:.1} max_recon_mb={:.1} max_residual_float_mb={:.1} max_residual_bucketized_mb={:.1} max_residual_bits_mb={:.1} max_residual_packed_mb={:.1} max_working_set_est_mb={:.1}",
            num_shards,
            mb(local_stats.max_emb_batch_bytes),
            mb(local_stats.max_score_est_bytes),
            mb(local_stats.max_code_batch_bytes),
            mb(local_stats.max_recon_bytes),
            mb(local_stats.max_residual_float_bytes),
            mb(local_stats.max_residual_bucketized_bytes),
            mb(local_stats.max_residual_bits_bytes),
            mb(local_stats.max_residual_packed_bytes),
            mb(local_stats.max_working_set_est_bytes)
        );
    }
    Ok(result)
}

pub fn compress_into_codes(embs: &Tensor, centroids: &Tensor) -> Tensor {
    let default_batch_sz = ((1 << 29) / centroids.size()[0].max(1)).max(1);
    compress_into_codes_with_batch(embs, centroids, default_batch_sz)
}

pub fn compress_into_codes_with_batch(embs: &Tensor, centroids: &Tensor, batch_sz: i64) -> Tensor {
    let embs = embs.to_kind(Kind::Half);
    let centroids = centroids.to_kind(Kind::Half);
    let mut codes = Vec::new();
    for mut emb_batch in embs.split(batch_sz.max(1), 0) {
        codes.push(centroids.matmul(&emb_batch.t_()).argmax(0, false));
    }
    Tensor::cat(&codes, 0)
}

pub fn packbits(res: &Tensor) -> Tensor {
    let bits_mat = res.reshape(&[-1, 8]);
    let weights = Tensor::from_slice(&BIT_WEIGHTS)
        .to_device(res.device())
        .to_kind(Kind::Float);
    let packed = bits_mat
        .to_kind(Kind::Float)
        .matmul(&weights)
        .to_kind(Kind::Uint8);
    packed
}
