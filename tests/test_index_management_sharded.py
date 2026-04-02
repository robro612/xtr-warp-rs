"""Sharded variants of index-management tests.

These tests mirror the core mutable-index behaviors from test_index_management.py
but run them on indices created with num_shards > 1.
"""

import json
import os
import shutil

import numpy as np
import pytest
import torch
from xtr_warp.search import XTRWarp

INDEX_DIR = ".indices/test_mgmt_sharded"
NUM_DOCS = 100
DOC_LEN = 128
DIM = 128
SEED = 42
NUM_SHARDS = 2

CREATE_KWARGS = dict(
    kmeans_niters=4,
    max_points_per_centroid=256,
    nbits=4,
    seed=SEED,
    device="cpu",
    num_shards=NUM_SHARDS,
)

SEARCH_KWARGS = dict(top_k=10, num_threads=1)


def _cleanup(index_name=INDEX_DIR):
    shutil.rmtree(index_name, ignore_errors=True)


def _fresh_index(index_name=INDEX_DIR, num_docs=NUM_DOCS):
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
    return {pid for query_res in results for pid, _score in query_res}


def _assert_is_sharded(index_name=INDEX_DIR, num_shards=NUM_SHARDS):
    meta = _load_metadata(index_name)
    assert meta["num_shards"] == num_shards
    assert meta["shard_boundaries"] is not None
    assert len(meta["shard_boundaries"]) == num_shards + 1
    for shard_id in range(num_shards):
        shard_dir = os.path.join(index_name, f"shard_{shard_id}")
        assert os.path.isdir(shard_dir), f"Missing shard directory {shard_dir}"


def test_sharded_delete():
    """Delete on sharded index should tombstone and filter from search."""
    idx, _docs, queries = _fresh_index()
    _assert_is_sharded()

    idx.load("cpu")
    results_before = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
    all_pids_before = _result_pids(results_before)
    assert all_pids_before

    target_pid = next(iter(all_pids_before))
    idx.delete([target_pid])

    results_after = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
    all_pids_after = _result_pids(results_after)
    assert target_pid not in all_pids_after
    assert os.path.exists(os.path.join(INDEX_DIR, "deleted_pids.npy"))

    _cleanup()


def test_sharded_add():
    """Add on sharded index should keep shard metadata and assign sequential IDs."""
    idx, _docs, queries = _fresh_index()
    _assert_is_sharded()

    meta_before = _load_metadata()
    num_before = meta_before["num_passages"]
    next_pid_before = meta_before.get("next_passage_id", num_before)

    new_docs = [torch.randn(DOC_LEN, DIM, device="cpu") for _ in range(20)]
    new_ids = idx.add(embeddings_source=new_docs, reload=False)

    assert len(new_ids) == 20
    assert new_ids == list(range(next_pid_before, next_pid_before + 20))

    meta_after = _load_metadata()
    assert meta_after["num_passages"] == num_before + 20
    assert meta_after["next_passage_id"] == next_pid_before + 20
    assert meta_after["num_shards"] == NUM_SHARDS
    assert len(meta_after["shard_boundaries"]) == NUM_SHARDS + 1

    idx.load("cpu")
    results = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
    assert len(results) == 5
    assert all(len(r) == 10 for r in results)

    _cleanup()


def test_sharded_delete_then_add_then_compact():
    """Delete/add lifecycle on sharded index should remain consistent after compact."""
    idx, _docs, queries = _fresh_index()
    _assert_is_sharded()

    meta_before = _load_metadata()
    next_pid = meta_before.get("next_passage_id", meta_before["num_passages"])

    # Tombstone some existing passages.
    to_delete = list(range(10))
    idx.delete(to_delete, compact_threshold=None)
    assert os.path.exists(os.path.join(INDEX_DIR, "deleted_pids.npy"))

    # Add new passages after delete.
    new_docs = [torch.randn(DOC_LEN, DIM, device="cpu") for _ in range(10)]
    new_ids = idx.add(embeddings_source=new_docs, reload=False)
    assert new_ids[0] == next_pid
    assert os.path.exists(os.path.join(INDEX_DIR, "deleted_pids.npy"))

    # Compact should clear tombstones while preserving sharded layout.
    idx.compact(reload=False)
    assert not os.path.exists(os.path.join(INDEX_DIR, "deleted_pids.npy"))

    meta_after = _load_metadata()
    expected_passages = NUM_DOCS - len(to_delete) + len(new_docs)
    assert meta_after["num_passages"] == expected_passages
    assert meta_after["num_embeddings"] == expected_passages * DOC_LEN
    assert meta_after["num_shards"] == NUM_SHARDS
    assert len(meta_after["shard_boundaries"]) == NUM_SHARDS + 1

    idx.load("cpu")
    results = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
    all_pids = _result_pids(results)
    for pid in to_delete:
        assert pid not in all_pids
    assert all(pid >= next_pid for pid in new_ids)

    _cleanup()


def test_sharded_add_with_reload():
    """add(reload=True) should keep sharded index immediately searchable."""
    idx, _docs, queries = _fresh_index()
    _assert_is_sharded()
    idx.load("cpu")

    new_docs = [torch.randn(DOC_LEN, DIM, device="cpu") for _ in range(8)]
    new_ids = idx.add(embeddings_source=new_docs)
    assert len(new_ids) == 8

    results = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
    assert len(results) == 5
    assert all(len(r) == 10 for r in results)

    _cleanup()


def test_sharded_update_preserves_ids():
    """update() should preserve ID watermark semantics on sharded index."""
    idx, _docs, queries = _fresh_index()
    _assert_is_sharded()
    meta_before = _load_metadata()

    target_pid = 5
    idx.update(
        passage_ids=[target_pid],
        embeddings_source=[torch.randn(DOC_LEN, DIM, device="cpu")],
        reload=False,
    )

    meta_after = _load_metadata()
    assert meta_after["next_passage_id"] == meta_before["next_passage_id"]
    assert meta_after["num_shards"] == NUM_SHARDS
    assert len(meta_after["shard_boundaries"]) == NUM_SHARDS + 1

    idx.load("cpu")
    results = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
    assert len(results) == 5
    assert all(len(r) == 10 for r in results)

    _cleanup()


def test_sharded_delete_idempotent():
    """Deleting the same PID twice should keep one tombstone entry."""
    idx, _docs, queries = _fresh_index()
    _assert_is_sharded()
    idx.load("cpu")

    idx.delete([5], compact_threshold=None)
    idx.delete([5], compact_threshold=None)

    pids = torch.from_numpy(np.load(os.path.join(INDEX_DIR, "deleted_pids.npy")))
    assert int((pids == 5).sum().item()) == 1

    results = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
    assert 5 not in _result_pids(results)

    _cleanup()


def test_sharded_auto_compact_on_delete():
    """delete(compact_threshold=...) should auto-compact for sharded indexes too."""
    idx, _docs, _queries = _fresh_index()
    _assert_is_sharded()
    tombstone_path = os.path.join(INDEX_DIR, "deleted_pids.npy")

    idx.delete(list(range(10)), compact_threshold=0.2)
    assert os.path.exists(tombstone_path)

    idx.delete(list(range(10, 25)), compact_threshold=0.2)
    assert not os.path.exists(tombstone_path)

    meta = _load_metadata()
    assert meta["num_passages"] == NUM_DOCS - 25
    assert meta["num_shards"] == NUM_SHARDS
    assert len(meta["shard_boundaries"]) == NUM_SHARDS + 1

    _cleanup()


def test_sharded_metadata_consistency_chain():
    """Metadata should remain coherent across add/delete/compact in sharded mode."""
    idx, _docs, _queries = _fresh_index()
    _assert_is_sharded()

    meta = _load_metadata()
    assert meta["num_passages"] == NUM_DOCS
    assert meta["next_passage_id"] == NUM_DOCS
    assert meta["num_embeddings"] == NUM_DOCS * DOC_LEN

    ids = idx.add(
        embeddings_source=[torch.randn(DOC_LEN, DIM, device="cpu") for _ in range(5)],
        reload=False,
    )
    assert len(ids) == 5

    meta = _load_metadata()
    assert meta["num_passages"] == NUM_DOCS + 5
    assert meta["next_passage_id"] == NUM_DOCS + 5
    assert meta["num_embeddings"] == (NUM_DOCS + 5) * DOC_LEN
    assert meta["num_shards"] == NUM_SHARDS

    idx.delete([0, 1, 2], compact_threshold=None)
    meta = _load_metadata()
    assert meta["num_passages"] == NUM_DOCS + 5
    assert meta["next_passage_id"] == NUM_DOCS + 5

    idx.compact(reload=False)
    meta = _load_metadata()
    assert meta["num_passages"] == NUM_DOCS + 5 - 3
    assert meta["next_passage_id"] == NUM_DOCS + 5
    assert meta["num_embeddings"] == (NUM_DOCS + 5 - 3) * DOC_LEN
    assert meta["num_shards"] == NUM_SHARDS
    assert len(meta["shard_boundaries"]) == NUM_SHARDS + 1

    _cleanup()


def test_sharded_load_device_rules():
    """Sharded load accepts one or N devices, rejects mismatched counts."""
    idx, _docs, _queries = _fresh_index()
    _assert_is_sharded()

    # Single device should be replicated across shards.
    idx.load("cpu")
    assert idx.devices == ["cpu", "cpu"]
    idx.free()

    # Explicit per-shard devices is accepted.
    idx.load(["cpu", "cpu"])
    assert idx.devices == ["cpu", "cpu"]
    idx.free()

    # Mismatched number of devices should fail.
    with pytest.raises(ValueError):
        idx.load(["cpu", "cpu", "cpu"])

    _cleanup()


def test_sharded_search_after_all_operations():
    """Full lifecycle smoke on sharded index."""
    idx, _docs, queries = _fresh_index()
    _assert_is_sharded()
    idx.load("cpu")

    # Initial search
    results = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
    assert len(results) == 5

    # Delete a document and ensure it's filtered
    idx.delete([0], compact_threshold=None)
    results = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
    assert 0 not in _result_pids(results)

    # Add new docs (reload=True default)
    new_docs = [torch.randn(DOC_LEN, DIM, device="cpu") for _ in range(5)]
    new_ids = idx.add(embeddings_source=new_docs)
    assert len(new_ids) == 5

    results = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
    assert len(results) == 5
    assert all(len(r) == 10 for r in results)

    # Update an existing document
    idx.update(
        passage_ids=[10],
        embeddings_source=[torch.randn(DOC_LEN, DIM, device="cpu")],
    )
    results = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
    assert len(results) == 5

    # Compact and verify state is still consistent/searchable
    idx.compact()
    results = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
    assert len(results) == 5
    assert all(len(r) == 10 for r in results)
    assert 0 not in _result_pids(results)

    meta = _load_metadata()
    assert meta["num_shards"] == NUM_SHARDS
    assert len(meta["shard_boundaries"]) == NUM_SHARDS + 1

    _cleanup()


def test_sharded_multiple_sequential_adds():
    """Sequential adds should keep IDs unique and metadata consistent in sharded mode."""
    idx, _docs, queries = _fresh_index()
    _assert_is_sharded()

    all_new_ids = []
    for _ in range(5):
        new_docs = [torch.randn(DOC_LEN, DIM, device="cpu") for _ in range(3)]
        new_ids = idx.add(embeddings_source=new_docs, reload=False)
        assert len(new_ids) == 3
        assert not (set(new_ids) & set(all_new_ids)), "IDs must not overlap"
        all_new_ids.extend(new_ids)

    meta = _load_metadata()
    assert meta["num_passages"] == NUM_DOCS + 15
    assert meta["next_passage_id"] == NUM_DOCS + 15
    assert meta["num_embeddings"] == (NUM_DOCS + 15) * DOC_LEN
    assert meta["num_shards"] == NUM_SHARDS
    assert len(meta["shard_boundaries"]) == NUM_SHARDS + 1

    idx.load("cpu")
    results = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
    assert len(results) == 5
    assert all(len(r) == 10 for r in results)

    _cleanup()

