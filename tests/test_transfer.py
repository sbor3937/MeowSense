"""Hermetic tests for the transfer-learning probe plumbing.

These exercise everything except the pretrained backbone itself: the embedding
cache round-trip, the probe construction, and the grouped-CV evaluation on
synthetic embeddings. They need scikit-learn but neither ``transformers``,
``torch``, nor the real dataset -- importing ``embeddings`` does not import the
backbone (that happens lazily inside ``_load_backbone``).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn", reason="scikit-learn required for the probe")

import embeddings as E  # noqa: E402
import train_transfer as TT  # noqa: E402


class TestCachePath:
    def test_slug_is_filesystem_safe(self):
        p = E._cache_path("/tmp/cache", "MIT/ast-finetuned-audioset-10-10-0.4593")
        assert "/" not in p.name
        assert p.suffix == ".npz"
        assert p.name.startswith("MIT__ast-finetuned-audioset")

    def test_distinct_models_distinct_files(self):
        a = E._cache_path("c", "org/model-a")
        b = E._cache_path("c", "org/model-b")
        assert a != b


class TestEmbeddingCacheRoundtrip:
    def test_loads_from_cache_without_backbone(self, tmp_path):
        """A present cache is returned verbatim, never touching the backbone."""
        model = "fake/model"
        X = np.random.default_rng(0).normal(size=(6, 8)).astype(np.float32)
        y = np.array(["B", "F", "I", "B", "F", "I"])
        groups = np.array(["C1", "C1", "C1", "C2", "C2", "C2"])

        cache_file = E._cache_path(tmp_path, model)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_file, X=X, y=y, groups=groups)

        # data_dir points nowhere; if the cache is honored it is never read.
        Xc, yc, gc = E.extract_embeddings(
            data_dir=tmp_path / "nonexistent",
            model_name=model,
            cache_dir=tmp_path,
            force=False,
            verbose=False,
        )
        assert np.allclose(Xc, X)
        assert list(yc) == list(y)
        assert list(gc) == list(groups)


class TestProbes:
    def test_build_probes_returns_two_pipelines(self):
        from sklearn.pipeline import Pipeline

        probes = TT.build_probes(seed=0)
        assert len(probes) == 2
        assert all(isinstance(p, Pipeline) for p in probes.values())
        # Every probe standardizes before classifying.
        assert all("scale" in dict(p.named_steps) for p in probes.values())

    def test_grouped_cv_recovers_separable_signal(self):
        """On embeddings that encode the label, the probe should score high and
        never leak a cat across the split."""
        rng = np.random.default_rng(1)
        classes = ["B", "F", "I"]
        X, y, groups = [], [], []
        for cat in range(9):
            for c_idx, c in enumerate(classes):
                for _ in range(5):
                    # Class-dependent mean => linearly separable; small noise.
                    vec = np.zeros(6) + c_idx * 4.0 + rng.normal(0, 0.3, size=6)
                    X.append(vec)
                    y.append(c)
                    groups.append(f"CAT{cat}")
        X, y, groups = np.array(X), np.array(y), np.array(groups)

        results = TT.evaluate_grouped_cv(TT.build_probes(0), X, y, groups, n_splits=3)

        assert set(results) == {"LogReg probe (AST emb.)", "SVM-RBF (AST emb.)"}
        for res in results.values():
            assert res["accuracies"].mean() > 0.9  # signal is trivially separable
            assert len(res["y_true"]) == len(y)  # every clip tested once
