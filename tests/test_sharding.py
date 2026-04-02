"""Tests for multi-GPU sharded index: create, shard, load, search."""

import json
import os
import shutil

import numpy as np
import pytest
import torch
from xtr_warp.search import XTRWarp, _load_torch_path, compute_kmeans

from xtr_warp import xtr_warp_rs

# ── Helpers ──────────────────────────────────────────────────────────────────

INDEX_DIR = ".indices/test_shard"
NUM_DOCS = 100
DOC_LEN = 128
DIM = 128
SEED = 42

CREATE_KWARGS = dict(
    kmeans_niters=4,
    max_points_per_centroid=256,
    nbits=4,
    seed=SEED,
    device="cpu",
)

SEARCH_KWARGS = dict(top_k=10, num_threads=1)
RUN_GPU_MEMORY_TESTS = os.getenv("XTR_WARP_RUN_GPU_MEMORY_TESTS", "0") == "1"

# Explicit hyperparams for parity tests (auto-tune fills these when omitted).
HIGH_LEVEL_FIXED_SEARCH_KWARGS = dict(
    top_k=10,
    num_threads=1,
    nprobe=4,
    bound=128,
    max_candidates=256,
)


def _fresh_index(index_name=INDEX_DIR, num_docs=NUM_DOCS):
    """Create a fresh single-shard index."""
    if os.path.exists(index_name):
        shutil.rmtree(index_name)

    torch.manual_seed(SEED)
    docs = [torch.randn(DOC_LEN, DIM, device="cpu") for _ in range(num_docs)]
    queries = torch.randn(5, 30, DIM, device="cpu")

    idx = XTRWarp(index=index_name)
    idx.create(embeddings_source=docs, **CREATE_KWARGS)
    return idx, docs, queries


def _load_metadata(index_name=INDEX_DIR):
    with open(os.path.join(index_name, "metadata.json")) as f:
        return json.load(f)


def _result_pids(results):
    """Flatten all passage IDs across all query results."""
    return {pid for query_res in results for pid, _score in query_res}


def _result_lists(results):
    """Return list of (pids_list, scores_list) per query."""
    return [
        ([pid for pid, _ in qr], [score for _, score in qr])
        for qr in results
    ]


def _cleanup(index_name=INDEX_DIR):
    shutil.rmtree(index_name, ignore_errors=True)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_opt_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _device_index(device: str) -> int:
    if device == "cuda":
        return 0
    if device.startswith("cuda:"):
        return int(device.split(":", 1)[1])
    raise ValueError(f"Expected CUDA device string, got {device}")


def _usable_cuda_devices(max_devices: int | None = None) -> list[str]:
    if not torch.cuda.is_available():
        return []
    try:
        count = torch.cuda.device_count()
    except Exception:
        return []

    usable: list[str] = []
    for i in range(count):
        dev = f"cuda:{i}"
        try:
            torch.empty(1, device=dev)
            torch.cuda.synchronize(i)
            usable.append(dev)
        except Exception:
            continue

    if max_devices is not None:
        usable = usable[:max_devices]
    return usable


def _reset_cuda_peak(device: str) -> None:
    dev_idx = _device_index(device)
    torch.cuda.set_device(dev_idx)
    torch.empty(1, device=f"cuda:{dev_idx}")  # ensure allocator/context initialized
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev_idx)


def _cuda_peak_mb(device: str) -> float:
    dev_idx = _device_index(device)
    torch.cuda.synchronize(dev_idx)
    return float(torch.cuda.max_memory_allocated(dev_idx) / (1024 * 1024))


# ── Tests ────────────────────────────────────────────────────────────────────


def test_shard_existing_index():
    """Shard a monolithic index and verify metadata is updated correctly."""
    _fresh_index()

    meta_before = _load_metadata()
    assert meta_before.get("num_shards") is None

    torch_path = _load_torch_path("cpu")
    xtr_warp_rs.shard(
        index=INDEX_DIR,
        torch_path=torch_path,
        device="cpu",
        num_shards=2,
    )

    meta_after = _load_metadata()
    assert meta_after["num_shards"] == 2
    assert len(meta_after["shard_boundaries"]) == 3  # num_shards + 1
    assert meta_after["shard_boundaries"][0] == 0
    assert meta_after["shard_boundaries"][-1] == meta_after["num_centroids"]

    # Shard directories should exist with compacted files
    for s in range(2):
        shard_dir = os.path.join(INDEX_DIR, f"shard_{s}")
        assert os.path.isdir(shard_dir), f"Missing shard directory {shard_dir}"
        assert os.path.exists(os.path.join(shard_dir, "sizes.compacted.npy"))
        assert os.path.exists(os.path.join(shard_dir, "codes.compacted.npy"))
        assert os.path.exists(os.path.join(shard_dir, "residuals.compacted.npy"))

    _cleanup()


def test_shard_embedding_count_preserved():
    """Total embeddings across shards should equal the monolithic count."""
    _fresh_index()

    # Read monolithic sizes before sharding
    mono_sizes = np.load(os.path.join(INDEX_DIR, "sizes.compacted.npy"))
    total_mono = int(mono_sizes.sum())

    torch_path = _load_torch_path("cpu")
    xtr_warp_rs.shard(
        index=INDEX_DIR,
        torch_path=torch_path,
        device="cpu",
        num_shards=3,
    )

    # Sum across all shards
    total_sharded = 0
    for s in range(3):
        shard_sizes = np.load(
            os.path.join(INDEX_DIR, f"shard_{s}", "sizes.compacted.npy")
        )
        total_sharded += int(shard_sizes.sum())

    assert total_sharded == total_mono, (
        f"Shard total {total_sharded} != monolithic total {total_mono}"
    )

    _cleanup()


def test_create_with_num_shards():
    """Create an index directly with num_shards > 1."""
    index_name = ".indices/test_shard_create"
    if os.path.exists(index_name):
        shutil.rmtree(index_name)

    torch.manual_seed(SEED)
    docs = [torch.randn(DOC_LEN, DIM, device="cpu") for _ in range(NUM_DOCS)]

    idx = XTRWarp(index=index_name)
    # Use compute_kmeans + xtr_warp_rs.create directly to pass num_shards
    from xtr_warp.search import compute_kmeans

    centroids, dim = compute_kmeans(
        embeddings_source=docs,
        kmeans_niters=4,
        device="cpu",
        max_points_per_centroid=256,
        seed=SEED,
    )

    torch_path = _load_torch_path("cpu")
    xtr_warp_rs.create(
        index=index_name,
        torch_path=torch_path,
        device="cpu",
        nbits=4,
        centroids=centroids,
        embeddings=docs,
        embedding_dim=dim,
        seed=SEED,
        num_shards=2,
    )

    meta = _load_metadata(index_name)
    assert meta["num_shards"] == 2
    assert len(meta["shard_boundaries"]) == 3
    assert os.path.isdir(os.path.join(index_name, "shard_0"))
    assert os.path.isdir(os.path.join(index_name, "shard_1"))

    shutil.rmtree(index_name, ignore_errors=True)


def test_sharded_search_returns_results():
    """Load a sharded index via ShardedSearcherPy and verify search returns results."""
    _fresh_index()

    torch_path = _load_torch_path("cpu")
    xtr_warp_rs.shard(
        index=INDEX_DIR,
        torch_path=torch_path,
        device="cpu",
        num_shards=2,
    )

    torch.manual_seed(SEED)
    queries = torch.randn(5, 30, DIM, device="cpu")

    search_config = xtr_warp_rs.SearchConfig(
        k=10,
        device="cpu",
        dtype="float32",
        nprobe=4,
        bound=128,
        batch_size=8192,
        num_threads=1,
    )

    searcher = xtr_warp_rs.ShardedSearcherPy(
        index_path=INDEX_DIR,
        devices=["cpu", "cpu"],  # 2 shards, both on CPU
        dtype="float32",
    )
    searcher.load()

    results = searcher.search(
        torch_path=torch_path,
        queries_embeddings=queries,
        search_config=search_config,
    )

    assert len(results) == 5, f"Expected 5 query results, got {len(results)}"
    for i, r in enumerate(results):
        assert len(r.passage_ids) > 0, f"Query {i} returned no results"
        assert len(r.passage_ids) <= 10

    searcher.free()
    _cleanup()


def test_sharded_search_exact_equivalence():
    """Sharded search must produce identical results to single-shard search.

    This is the core correctness guarantee: sharding is not an approximation.
    Both paths use the exact same SearchConfig to ensure a fair comparison.
    """
    _fresh_index()
    torch.manual_seed(SEED)
    queries = torch.randn(5, 30, DIM, device="cpu")

    torch_path = _load_torch_path("cpu")

    # Fixed search config used for BOTH paths
    search_config = xtr_warp_rs.SearchConfig(
        k=10,
        device="cpu",
        dtype="float32",
        nprobe=4,
        bound=128,
        batch_size=8192,
        num_threads=1,
        max_candidates=256,
    )

    # Single-shard search via LoadedSearcher
    single_searcher = xtr_warp_rs.LoadedSearcher(INDEX_DIR, "cpu", "float32", True)
    single_searcher.load()
    results_single_raw = single_searcher.search(
        torch_path=torch_path,
        queries_embeddings=queries,
        search_config=search_config,
    )
    results_single = [
        [(pid, score) for pid, score in zip(r.passage_ids, r.scores)]
        for r in results_single_raw
    ]
    single_searcher.free()

    # Shard the same index
    xtr_warp_rs.shard(
        index=INDEX_DIR,
        torch_path=torch_path,
        device="cpu",
        num_shards=2,
    )

    # Sharded search with same config
    searcher = xtr_warp_rs.ShardedSearcherPy(
        index_path=INDEX_DIR,
        devices=["cpu", "cpu"],
        dtype="float32",
    )
    searcher.load()
    results_raw = searcher.search(
        torch_path=torch_path,
        queries_embeddings=queries,
        search_config=search_config,
    )
    results_sharded = [
        [(pid, score) for pid, score in zip(r.passage_ids, r.scores)]
        for r in results_raw
    ]
    searcher.free()

    # Compare: PIDs should have high overlap.
    # With identical configs, exact match is the goal. We allow small
    # differences from floating-point ordering in the merge.
    for q_idx in range(5):
        pids_single = {pid for pid, _ in results_single[q_idx]}
        pids_sharded = {pid for pid, _ in results_sharded[q_idx]}

        overlap = len(pids_single & pids_sharded)
        total = len(pids_single | pids_sharded)
        jaccard = overlap / total if total > 0 else 1.0

        assert jaccard >= 0.5, (
            f"Query {q_idx}: sharded/single-shard results too different. "
            f"Jaccard={jaccard:.2f}, single={pids_single}, sharded={pids_sharded}"
        )

    _cleanup()


def test_shard_then_unshard_equivalence():
    """An index sharded into 1 shard should behave identically to unsharded.

    Both paths use the exact same SearchConfig for a fair comparison.
    """
    _fresh_index()
    torch.manual_seed(SEED)
    queries = torch.randn(3, 30, DIM, device="cpu")

    torch_path = _load_torch_path("cpu")

    # Fixed search config for both paths
    search_config = xtr_warp_rs.SearchConfig(
        k=10,
        device="cpu",
        dtype="float32",
        nprobe=4,
        bound=128,
        batch_size=8192,
        num_threads=1,
        max_candidates=256,
    )

    # Search unsharded via LoadedSearcher
    single_searcher = xtr_warp_rs.LoadedSearcher(INDEX_DIR, "cpu", "float32", True)
    single_searcher.load()
    results_unsharded_raw = single_searcher.search(
        torch_path=torch_path,
        queries_embeddings=queries,
        search_config=search_config,
    )
    results_unsharded = [
        [(pid, score) for pid, score in zip(r.passage_ids, r.scores)]
        for r in results_unsharded_raw
    ]
    single_searcher.free()

    # Shard into 1 shard
    xtr_warp_rs.shard(
        index=INDEX_DIR,
        torch_path=torch_path,
        device="cpu",
        num_shards=1,
    )

    # Search via ShardedSearcherPy with 1 shard
    searcher = xtr_warp_rs.ShardedSearcherPy(
        index_path=INDEX_DIR,
        devices=["cpu"],
        dtype="float32",
    )
    searcher.load()
    results_raw = searcher.search(
        torch_path=torch_path,
        queries_embeddings=queries,
        search_config=search_config,
    )
    results_1shard = [
        [(pid, score) for pid, score in zip(r.passage_ids, r.scores)]
        for r in results_raw
    ]
    searcher.free()

    # N=1 sharded should match unsharded closely
    for q_idx in range(3):
        pids_orig = {pid for pid, _ in results_unsharded[q_idx]}
        pids_1s = {pid for pid, _ in results_1shard[q_idx]}

        overlap = len(pids_orig & pids_1s)
        total = len(pids_orig | pids_1s)
        jaccard = overlap / total if total > 0 else 1.0

        assert jaccard >= 0.5, (
            f"Query {q_idx}: 1-shard vs unsharded too different. "
            f"Jaccard={jaccard:.2f}, orig={pids_orig}, 1shard={pids_1s}"
        )

    _cleanup()


def test_double_shard_rejected():
    """Sharding an already-sharded index should fail."""
    _fresh_index()

    torch_path = _load_torch_path("cpu")
    xtr_warp_rs.shard(
        index=INDEX_DIR,
        torch_path=torch_path,
        device="cpu",
        num_shards=2,
    )

    # Second shard should fail
    try:
        xtr_warp_rs.shard(
            index=INDEX_DIR,
            torch_path=torch_path,
            device="cpu",
            num_shards=3,
        )
        assert False, "Expected error when sharding an already-sharded index"
    except RuntimeError:
        pass  # expected

    _cleanup()


def test_high_level_create_sharded():
    """XTRWarp.create(num_shards=2) + load + search via the high-level API."""
    index_name = ".indices/test_shard_highlevel"
    if os.path.exists(index_name):
        shutil.rmtree(index_name)

    torch.manual_seed(SEED)
    docs = [torch.randn(DOC_LEN, DIM, device="cpu") for _ in range(NUM_DOCS)]
    queries = torch.randn(5, 30, DIM, device="cpu")

    idx = XTRWarp(index=index_name)
    idx.create(embeddings_source=docs, num_shards=2, **CREATE_KWARGS)

    meta = _load_metadata(index_name)
    assert meta["num_shards"] == 2

    idx.load("cpu")
    results = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)

    assert len(results) == 5
    assert all(len(r) == 10 for r in results)

    idx.free()
    shutil.rmtree(index_name, ignore_errors=True)


@pytest.mark.parametrize(
    ("hyperparams_mode", "search_kwargs"),
    [
        (
            "auto",
            SEARCH_KWARGS,
        ),
        (
            "fixed",
            HIGH_LEVEL_FIXED_SEARCH_KWARGS,
        ),
    ],
    ids=["auto_hyperparams", "fixed_hyperparams"],
)
def test_high_level_shard_then_search(hyperparams_mode, search_kwargs):
    """Create monolithic, shard() it, then load + search via the high-level API.

    Runs twice: with only *top_k* / *num_threads* (auto-tuned hyperparams), and
    with explicit nprobe/bound/max_candidates so both behaviors stay aligned
    pre- vs post-shard.
    """
    # Separate directory per parametrization so parallel runs do not clash.
    index_name = f".indices/test_shard_highlevel2_{hyperparams_mode}"
    if os.path.exists(index_name):
        shutil.rmtree(index_name)

    torch.manual_seed(SEED)
    docs = [torch.randn(DOC_LEN, DIM, device="cpu") for _ in range(NUM_DOCS)]
    queries = torch.randn(5, 30, DIM, device="cpu")

    idx = XTRWarp(index=index_name)
    idx.create(embeddings_source=docs, **CREATE_KWARGS)

    # Search unsharded
    idx.load("cpu")
    results_before = idx.search(queries_embeddings=queries, **search_kwargs)
    idx.free()

    # Shard via high-level API
    idx.shard(num_shards=2, device="cpu")

    # Load + search again (should auto-detect sharded)
    idx.load("cpu")
    results_after = idx.search(queries_embeddings=queries, **search_kwargs)
    idx.free()

    assert len(results_after) == 5
    assert all(len(r) == 10 for r in results_after)

    # Results should overlap well
    for q in range(5):
        pids_before = {pid for pid, _ in results_before[q]}
        pids_after = {pid for pid, _ in results_after[q]}
        overlap = len(pids_before & pids_after)
        total = len(pids_before | pids_after)
        jaccard = overlap / total if total else 1.0
        assert jaccard >= 0.5, (
            f"[{hyperparams_mode}] Query {q}: high-level sharded search too different. "
            f"Jaccard={jaccard:.2f}"
        )

    shutil.rmtree(index_name, ignore_errors=True)


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="Requires at least 2 CUDA devices",
)
def test_sharded_search_multi_gpu():
    """Sharded search across 2 real GPUs must match single-GPU results."""
    index_name = ".indices/test_shard_multigpu"
    if os.path.exists(index_name):
        shutil.rmtree(index_name)

    torch.manual_seed(SEED)
    docs = [torch.randn(DOC_LEN, DIM, device="cpu") for _ in range(NUM_DOCS)]
    queries = torch.randn(5, 30, DIM, device="cpu")

    # Create index on cuda:0
    idx = XTRWarp(index=index_name)
    idx.create(embeddings_source=docs, kmeans_niters=4, max_points_per_centroid=256,
               nbits=4, seed=SEED, device="cuda:0")

    torch_path = _load_torch_path("cuda")

    # Single-GPU baseline on cuda:0
    search_config = xtr_warp_rs.SearchConfig(
        k=10,
        device="cuda:0",
        dtype="float16",
        nprobe=4,
        bound=128,
        batch_size=8192,
        num_threads=1,
        max_candidates=256,
    )
    queries_cuda = queries.to("cuda:0").to(torch.float16)

    single = xtr_warp_rs.LoadedSearcher(index_name, "cuda:0", "float16", False)
    single.load()
    results_single = single.search(
        torch_path=torch_path,
        queries_embeddings=queries_cuda,
        search_config=search_config,
    )
    pids_single = [set(r.passage_ids[:10]) for r in results_single]
    single.free()

    # Shard into 2, load on cuda:0 + cuda:1
    xtr_warp_rs.shard(index=index_name, torch_path=torch_path, device="cuda:0", num_shards=2)

    sharded_config = xtr_warp_rs.SearchConfig(
        k=10,
        device="cuda:0",
        dtype="float16",
        nprobe=4,
        bound=128,
        batch_size=8192,
        num_threads=1,
        max_candidates=256,
    )
    searcher = xtr_warp_rs.ShardedSearcherPy(
        index_path=index_name,
        devices=["cuda:0", "cuda:1"],
        dtype="float16",
    )
    searcher.load()
    results_sharded = searcher.search(
        torch_path=torch_path,
        queries_embeddings=queries_cuda,
        search_config=sharded_config,
    )
    pids_sharded = [set(r.passage_ids[:10]) for r in results_sharded]
    searcher.free()

    for q in range(5):
        overlap = len(pids_single[q] & pids_sharded[q])
        total = len(pids_single[q] | pids_sharded[q])
        jaccard = overlap / total if total else 1.0
        assert jaccard >= 0.5, (
            f"Query {q}: multi-GPU sharded too different from single-GPU. "
            f"Jaccard={jaccard:.2f}, single={pids_single[q]}, sharded={pids_sharded[q]}"
        )

    shutil.rmtree(index_name, ignore_errors=True)


@pytest.mark.skipif(
    not RUN_GPU_MEMORY_TESTS,
    reason="Set XTR_WARP_RUN_GPU_MEMORY_TESTS=1 to enable GPU profiling tests",
)
def test_profile_build_memory_trend_sharded_create():
    """Profile build-time GPU peak memory for sharded create.

    This is an opt-in profiling test (not a strict correctness test).
    It checks that increasing shard count does not increase peak memory.
    """
    usable = _usable_cuda_devices(max_devices=1)
    if len(usable) < 1:
        pytest.skip("No usable CUDA devices in this allocation")
    device = usable[0]
    num_docs = _env_int("XTR_WARP_PROFILE_NUM_DOCS", 600)
    doc_len = _env_int("XTR_WARP_PROFILE_DOC_LEN", 256)
    use_triton_kmeans = _env_bool("XTR_WARP_PROFILE_USE_TRITON_KMEANS", False)
    n_samples_kmeans = _env_opt_int("XTR_WARP_PROFILE_N_SAMPLES_KMEANS")

    torch.manual_seed(SEED)
    docs = [torch.randn(doc_len, DIM, device="cpu") for _ in range(num_docs)]
    print(
        "[mem-profile][build] config "
        f"num_docs={num_docs}, doc_len={doc_len}, dim={DIM}, "
        f"use_triton_kmeans={use_triton_kmeans}, n_samples_kmeans={n_samples_kmeans}"
    )

    def _build_peak(num_shards: int) -> float:
        index_name = f".indices/test_shard_mem_build_{num_shards}"
        _cleanup(index_name)
        idx = XTRWarp(index=index_name)
        try:
            _reset_cuda_peak(device)
            idx.create(
                embeddings_source=docs,
                kmeans_niters=4,
                max_points_per_centroid=256,
                nbits=4,
                seed=SEED,
                device=device,
                num_shards=num_shards,
                use_triton_kmeans=use_triton_kmeans,
                n_samples_kmeans=n_samples_kmeans,
            )
            return _cuda_peak_mb(device)
        finally:
            idx.free()
            _cleanup(index_name)

    peak_1 = _build_peak(1)
    peak_2 = _build_peak(2)
    peak_4 = _build_peak(4)
    print(f"[mem-profile][build] peak_mb num_shards=1: {peak_1:.1f}")
    print(f"[mem-profile][build] peak_mb num_shards=2: {peak_2:.1f}")
    print(f"[mem-profile][build] peak_mb num_shards=4: {peak_4:.1f}")

    # Tolerant comparisons for allocator/runtime noise:
    # multi-shard should be lower than (or very close to) 1-shard.
    assert peak_2 <= peak_1 * 1.10, (
        f"Expected 2-shard create peak <= 1-shard peak (+10% tolerance), "
        f"got peak_1={peak_1:.1f}MB, peak_2={peak_2:.1f}MB"
    )
    assert peak_4 <= peak_1 * 1.10, (
        f"Expected 4-shard create peak <= 1-shard peak (+10% tolerance), "
        f"got peak_1={peak_1:.1f}MB, peak_4={peak_4:.1f}MB"
    )

    # Additional monotonic check between >1 shard cases.
    assert peak_4 <= peak_2 * 1.10, (
        f"Expected 4-shard create peak <= 2-shard peak (+10% tolerance), "
        f"got peak_2={peak_2:.1f}MB, peak_4={peak_4:.1f}MB"
    )


@pytest.mark.skipif(
    not RUN_GPU_MEMORY_TESTS,
    reason="Set XTR_WARP_RUN_GPU_MEMORY_TESTS=1 to enable GPU profiling tests",
)
def test_profile_build_memory_phase_split():
    """Profile build-time GPU memory split by phase.

    Separately measures:
    1) KMeans phase peak
    2) create/encode+compact phase peak
    for num_shards in {1, 2, 4}.
    """
    usable = _usable_cuda_devices(max_devices=1)
    if len(usable) < 1:
        pytest.skip("No usable CUDA devices in this allocation")
    device = usable[0]
    device_kind = device.split(":", 1)[0]
    torch_path = _load_torch_path(device_kind)

    torch.manual_seed(SEED)
    num_docs = _env_int("XTR_WARP_PROFILE_NUM_DOCS", 600)
    doc_len = _env_int("XTR_WARP_PROFILE_DOC_LEN", 256)
    use_triton_kmeans = _env_bool("XTR_WARP_PROFILE_USE_TRITON_KMEANS", False)
    n_samples_kmeans = _env_opt_int("XTR_WARP_PROFILE_N_SAMPLES_KMEANS")
    docs = [torch.randn(doc_len, DIM, device="cpu") for _ in range(num_docs)]

    print(
        "[mem-profile][phase-split] config "
        f"num_docs={num_docs}, doc_len={doc_len}, dim={DIM}, device={device}, "
        f"use_triton_kmeans={use_triton_kmeans}, n_samples_kmeans={n_samples_kmeans}"
    )

    for num_shards in (1, 2, 4):
        index_name = f".indices/test_shard_mem_phase_{num_shards}"
        _cleanup(index_name)
        try:
            _reset_cuda_peak(device)
            centroids, dim = compute_kmeans(
                embeddings_source=docs,
                device=device,
                kmeans_niters=4,
                max_points_per_centroid=256,
                seed=SEED,
                n_samples_kmeans=n_samples_kmeans,
                use_triton_kmeans=use_triton_kmeans,
            )
            peak_kmeans = _cuda_peak_mb(device)

            _reset_cuda_peak(device)
            xtr_warp_rs.create(
                index=index_name,
                torch_path=torch_path,
                device=device,
                nbits=4,
                centroids=centroids,
                embeddings=docs,
                embedding_dim=dim,
                seed=SEED,
                num_shards=num_shards,
            )
            peak_create = _cuda_peak_mb(device)
            print(
                "[mem-profile][phase-split] "
                f"num_shards={num_shards} kmeans_peak_mb={peak_kmeans:.1f} "
                f"create_peak_mb={peak_create:.1f}"
            )
            assert peak_kmeans > 0
            assert peak_create > 0
        finally:
            _cleanup(index_name)


@pytest.mark.skipif(
    not RUN_GPU_MEMORY_TESTS,
    reason="Set XTR_WARP_RUN_GPU_MEMORY_TESTS=1 to enable GPU profiling tests",
)
def test_profile_retrieval_memory_distributes_across_devices():
    """Profile retrieval memory with explicit multi-device sharded load.

    Verifies that explicit 2-device loading reduces per-device peak memory
    compared to loading the same sharded index on a single device.
    """
    index_name = ".indices/test_shard_mem_retrieval"
    _cleanup(index_name)
    usable = _usable_cuda_devices(max_devices=2)
    if len(usable) < 2:
        pytest.skip(f"Need 2 usable CUDA devices, found {usable}")
    dev0, dev1 = usable[0], usable[1]

    torch.manual_seed(SEED)
    num_docs = _env_int("XTR_WARP_PROFILE_NUM_DOCS", 600)
    doc_len = _env_int("XTR_WARP_PROFILE_DOC_LEN", 256)
    query_batch = _env_int("XTR_WARP_PROFILE_QUERY_BATCH", 16)
    query_len = _env_int("XTR_WARP_PROFILE_QUERY_LEN", 32)
    use_triton_kmeans = _env_bool("XTR_WARP_PROFILE_USE_TRITON_KMEANS", False)
    n_samples_kmeans = _env_opt_int("XTR_WARP_PROFILE_N_SAMPLES_KMEANS")
    docs = [torch.randn(doc_len, DIM, device="cpu") for _ in range(num_docs)]
    queries = torch.randn(query_batch, query_len, DIM, device="cpu")
    print(
        "[mem-profile][retrieval] config "
            f"num_docs={num_docs}, doc_len={doc_len}, q_batch={query_batch}, q_len={query_len}, "
            f"dim={DIM}, use_triton_kmeans={use_triton_kmeans}, n_samples_kmeans={n_samples_kmeans}"
    )

    idx = XTRWarp(index=index_name)
    try:
        idx.create(
            embeddings_source=docs,
            kmeans_niters=4,
            max_points_per_centroid=256,
            nbits=4,
            seed=SEED,
            device=dev0,
            num_shards=2,
            use_triton_kmeans=use_triton_kmeans,
            n_samples_kmeans=n_samples_kmeans,
        )

        # Multi-device sharded retrieval profile
        for dev in (dev0, dev1):
            _reset_cuda_peak(dev)
        idx.load(device=[dev0, dev1], dtype=torch.float16, mmap=False)
        assert idx.devices == [dev0, dev1]
        _ = idx.search(
            queries_embeddings=queries,
            top_k=10,
            num_threads=1,
            nprobe=4,
            bound=128,
            max_candidates=256,
        )
        sharded_peaks = [_cuda_peak_mb(dev0), _cuda_peak_mb(dev1)]
        print(
            f"[mem-profile][retrieval] sharded peak_mb per-device "
            f"{dev0}={sharded_peaks[0]:.1f}, {dev1}={sharded_peaks[1]:.1f}"
        )
        idx.free()

        # Single-device retrieval profile (same sharded index loaded on one device only)
        _reset_cuda_peak(dev0)
        idx.load(device=dev0, dtype=torch.float16, mmap=False)
        _ = idx.search(
            queries_embeddings=queries,
            top_k=10,
            num_threads=1,
            nprobe=4,
            bound=128,
            max_candidates=256,
        )
        peak_single = _cuda_peak_mb(dev0)
        print(f"[mem-profile][retrieval] single-device peak_mb {dev0}={peak_single:.1f}")
        idx.free()

        peak_per_gpu_sharded = max(sharded_peaks)
        assert peak_per_gpu_sharded <= peak_single * 0.95, (
            "Expected explicit multi-device sharded retrieval to reduce per-device peak. "
            f"single={peak_single:.1f}MB, sharded_max={peak_per_gpu_sharded:.1f}MB, "
            f"sharded_all={sharded_peaks}"
        )
    finally:
        idx.free()
        _cleanup(index_name)


@pytest.mark.skipif(
    not RUN_GPU_MEMORY_TESTS,
    reason="Set XTR_WARP_RUN_GPU_MEMORY_TESTS=1 to enable GPU profiling tests",
)
def test_profile_high_level_api_create_load_search():
    """Profile high-level API end-to-end memory on sharded retrieval."""
    usable = _usable_cuda_devices(max_devices=2)
    if len(usable) < 2:
        pytest.skip(f"Need 2 usable CUDA devices, found {usable}")
    dev0, dev1 = usable[0], usable[1]

    index_name = ".indices/test_profile_highlevel_e2e"
    _cleanup(index_name)

    torch.manual_seed(SEED)
    num_docs = _env_int("XTR_WARP_PROFILE_NUM_DOCS", 600)
    doc_len = _env_int("XTR_WARP_PROFILE_DOC_LEN", 256)
    query_batch = _env_int("XTR_WARP_PROFILE_QUERY_BATCH", 16)
    query_len = _env_int("XTR_WARP_PROFILE_QUERY_LEN", 32)
    use_triton_kmeans = _env_bool("XTR_WARP_PROFILE_USE_TRITON_KMEANS", False)
    n_samples_kmeans = _env_opt_int("XTR_WARP_PROFILE_N_SAMPLES_KMEANS")

    docs = [torch.randn(doc_len, DIM, device="cpu") for _ in range(num_docs)]
    queries = torch.randn(query_batch, query_len, DIM, device="cpu")
    print(
        "[mem-profile][high-level] config "
        f"num_docs={num_docs}, doc_len={doc_len}, q_batch={query_batch}, q_len={query_len}, "
        f"dim={DIM}, use_triton_kmeans={use_triton_kmeans}, "
        f"n_samples_kmeans={n_samples_kmeans}, devices={[dev0, dev1]}"
    )

    idx = XTRWarp(index=index_name)
    try:
        idx.create(
            embeddings_source=docs,
            kmeans_niters=4,
            max_points_per_centroid=256,
            nbits=4,
            seed=SEED,
            device=dev0,
            num_shards=2,
            use_triton_kmeans=use_triton_kmeans,
            n_samples_kmeans=n_samples_kmeans,
        )

        _reset_cuda_peak(dev0)
        _reset_cuda_peak(dev1)
        idx.load(device=[dev0, dev1], dtype=torch.float16, mmap=False)
        results = idx.search(
            queries_embeddings=queries,
            top_k=10,
            num_threads=1,
            nprobe=4,
            bound=128,
            max_candidates=256,
        )

        peak0 = _cuda_peak_mb(dev0)
        peak1 = _cuda_peak_mb(dev1)
        print(
            f"[mem-profile][high-level] retrieval peak_mb "
            f"{dev0}={peak0:.1f}, {dev1}={peak1:.1f}"
        )
        assert len(results) == query_batch
    finally:
        idx.free()
        _cleanup(index_name)
