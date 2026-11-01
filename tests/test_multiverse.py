"""Multiverse manager tests: creation, isolation, checkpoints, branching, persistence."""
import os
import shutil
import tempfile
import numpy as np
import pytest

from universe_engine.multiverse import (
    MultiverseManager, UniverseRecord, CheckpointRecord,
    classify_cosmology, validate_cosmology, MultiverseError, REFERENCE_COSMOLOGY,
)


def tmp_root():
    d = tempfile.mkdtemp(prefix="multiverse-test-")
    return d


def test_reference_classification():
    assert classify_cosmology(dict(REFERENCE_COSMOLOGY)) == "observational"


def test_hypothetical_classification():
    c = dict(REFERENCE_COSMOLOGY, Omega_m=0.5, Omega_Lambda=0.5)
    assert classify_cosmology(c) == "hypothetical"


def test_outside_domain_classification():
    c = dict(REFERENCE_COSMOLOGY, Omega_m=2.0, Omega_Lambda=-1.0)
    assert classify_cosmology(c) == "outside_validated_domain"


def test_validate_rejects_nonflat():
    c = dict(REFERENCE_COSMOLOGY, Omega_Lambda=0.6)
    with pytest.raises(MultiverseError):
        validate_cosmology(c)


def test_validate_allows_future():
    c = dict(REFERENCE_COSMOLOGY, a_end=5.0)
    validate_cosmology(c)  # must not raise


def test_validate_rejects_beyond_domain():
    c = dict(REFERENCE_COSMOLOGY, a_end=11.0)
    with pytest.raises(MultiverseError):
        validate_cosmology(c)


def test_create_and_list():
    root = tmp_root()
    try:
        m = MultiverseManager(root)
        r1 = m.create_universe("Alpha", seed=42)
        r2 = m.create_universe("Beta", seed=99)
        ids = {u.uuid for u in m.list_universes()}
        assert r1.uuid in ids and r2.uuid in ids
        names = {u.name for u in m.list_universes()}
        assert names == {"Alpha", "Beta"}
        # immutable UUIDs
        assert isinstance(r1.uuid, str) and len(r1.uuid) > 0
        assert r1.created_at and r1.status == "created"
    finally:
        shutil.rmtree(root)


def test_create_validates_params():
    root = tmp_root()
    try:
        m = MultiverseManager(root)
        with pytest.raises(MultiverseError):
            m.create_universe("")
        with pytest.raises(MultiverseError):
            m.create_universe("x" * 200)
        with pytest.raises(MultiverseError):
            m.create_universe("Bad", cosmology=dict(REFERENCE_COSMOLOGY, Omega_Lambda=0.6))
    finally:
        shutil.rmtree(root)


def test_isolation_different_seeds():
    root = tmp_root()
    try:
        m = MultiverseManager(root)
        a = m.create_universe("A", seed=1)
        b = m.create_universe("B", seed=2)
        assert a.seed != b.seed
        assert a.uuid != b.uuid
        # different records, independent mutable state
        m.set_time(a.uuid, 1e17, 2.0)
        assert m.get(a.uuid).cosmic_time_s == 1e17
        assert m.get(b.uuid).cosmic_time_s == 0.0
    finally:
        shutil.rmtree(root)


def test_same_seed_reproduces():
    root = tmp_root()
    try:
        m = MultiverseManager(root)
        c = dict(REFERENCE_COSMOLOGY, a_end=2.0, steps=100)
        r1 = m.create_universe("Rep1", seed=7, cosmology=c)
        r2 = m.create_universe("Rep2", seed=7, cosmology=c)
        assert r1.cosmology == r2.cosmology
        assert r1.seed == r2.seed
        assert r1.model_settings == r2.model_settings
    finally:
        shutil.rmtree(root)


def test_rename_and_delete():
    root = tmp_root()
    try:
        m = MultiverseManager(root)
        r = m.create_universe("Old")
        m.rename(r.uuid, "New")
        assert m.get(r.uuid).name == "New"
        m.delete(r.uuid)
        with pytest.raises(MultiverseError):
            m.get(r.uuid)
        assert len(m.list_universes()) == 0
    finally:
        shutil.rmtree(root)


def test_checkpoint_roundtrip():
    root = tmp_root()
    try:
        m = MultiverseManager(root)
        r = m.create_universe("CP")
        m.set_time(r.uuid, 1e17, 2.0)
        arrays = {"x": np.arange(5, dtype=np.float64), "v": np.zeros(5)}
        cp = m.checkpoint(r.uuid, cid="cp1", arrays=arrays)
        assert cp.id == "cp1"
        assert cp.cosmic_time_s == 1e17
        assert len(m.list_checkpoints(r.uuid)) == 1
        # restore
        restored = m.restore_checkpoint(r.uuid, "cp1")
        assert restored["meta"]["cosmic_time_s"] == 1e17
        assert "x" in restored["arrays"]
        np.testing.assert_array_equal(restored["arrays"]["x"], np.arange(5, dtype=np.float64))
        # missing checkpoint
        with pytest.raises(MultiverseError):
            m.restore_checkpoint(r.uuid, "nope")
    finally:
        shutil.rmtree(root)


def test_persistence_reload():
    root = tmp_root()
    try:
        m1 = MultiverseManager(root)
        r = m1.create_universe("Persist", seed=123)
        m1.set_time(r.uuid, 5e16, 1.5)
        m1.checkpoint(r.uuid, cid="persist-cp", arrays={"g": np.array([1.0, 2.0])})
        uid = r.uuid

        # new manager instance reading same directory -> persisted state
        m2 = MultiverseManager(root)
        got = m2.get(uid)
        assert got.name == "Persist"
        assert got.seed == 123
        assert got.cosmic_time_s == 5e16
        assert got.scale_factor == 1.5
        assert len(got.checkpoints) == 1
        assert got.checkpoints[0]["id"] == "persist-cp"
    finally:
        shutil.rmtree(root)


def test_branch():
    root = tmp_root()
    try:
        m = MultiverseManager(root)
        r = m.create_universe("Parent", seed=5)
        m.set_time(r.uuid, 1e17, 2.0)
        m.checkpoint(r.uuid, cid="branch-source")
        b = m.branch(r.uuid, "Child", from_checkpoint="branch-source", seed=5)
        assert b.name == "Child"
        assert b.seed == 5
        assert b.cosmic_time_s == 1e17
        assert b.scale_factor == 2.0
        assert b.uuid != r.uuid
        # parent unaffected
        assert m.get(r.uuid).name == "Parent"
    finally:
        shutil.rmtree(root)


def test_checkpoint_limit():
    root = tmp_root()
    try:
        m = MultiverseManager(root)
        r = m.create_universe("Limit")
        for i in range(64):
            m.checkpoint(r.uuid, cid=f"cp-{i}")
        with pytest.raises(MultiverseError):
            m.checkpoint(r.uuid, cid="cp-65")
    finally:
        shutil.rmtree(root)


def test_import_no_unpickle():
    root = tmp_root()
    try:
        m = MultiverseManager(root)
        r = m.create_universe("Safe")
        cp = m.checkpoint(r.uuid, cid="safe", arrays={"a": np.array([1.0])})
        restored = m.restore_checkpoint(r.uuid, "safe")
        assert restored["meta"]["schema"] == "checkpoint/v1"
        assert restored["arrays"]["a"][0] == 1.0
    finally:
        shutil.rmtree(root)
