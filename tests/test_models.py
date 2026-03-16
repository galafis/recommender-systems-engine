"""
Unit tests for recommender systems engine models.
"""

import pytest
import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.recommender import (
    SimilarityMetrics,
    CollaborativeFilter,
    ContentBasedFilter,
    HybridRecommender,
    RecommenderEvaluator,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_interactions():
    """Sample user-item interactions for testing."""
    return [
        {"user_id": "u1", "item_id": "i1", "rating": 5.0},
        {"user_id": "u1", "item_id": "i2", "rating": 4.0},
        {"user_id": "u1", "item_id": "i3", "rating": 3.0},
        {"user_id": "u2", "item_id": "i1", "rating": 4.0},
        {"user_id": "u2", "item_id": "i2", "rating": 5.0},
        {"user_id": "u2", "item_id": "i4", "rating": 4.0},
        {"user_id": "u3", "item_id": "i1", "rating": 3.0},
        {"user_id": "u3", "item_id": "i3", "rating": 4.0},
        {"user_id": "u3", "item_id": "i5", "rating": 5.0},
        {"user_id": "u4", "item_id": "i2", "rating": 3.0},
        {"user_id": "u4", "item_id": "i4", "rating": 5.0},
        {"user_id": "u4", "item_id": "i5", "rating": 4.0},
    ]


@pytest.fixture
def sample_item_features():
    """Sample item feature vectors for content-based filtering."""
    return {
        "i1": {"action": 0.9, "comedy": 0.1, "drama": 0.3},
        "i2": {"action": 0.2, "comedy": 0.8, "drama": 0.5},
        "i3": {"action": 0.7, "comedy": 0.0, "drama": 0.8},
        "i4": {"action": 0.1, "comedy": 0.9, "drama": 0.2},
        "i5": {"action": 0.6, "comedy": 0.3, "drama": 0.9},
        "i6": {"action": 0.8, "comedy": 0.2, "drama": 0.4},
        "i7": {"action": 0.0, "comedy": 0.7, "drama": 0.6},
    }


# ---------------------------------------------------------------------------
# SimilarityMetrics Tests
# ---------------------------------------------------------------------------

class TestSimilarityMetrics:
    """Tests for similarity computation."""

    def test_cosine_identical_vectors(self):
        vec = {"a": 1.0, "b": 2.0, "c": 3.0}
        sim = SimilarityMetrics.cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 1e-6

    def test_cosine_orthogonal_vectors(self):
        vec_a = {"a": 1.0, "b": 0.0}
        vec_b = {"a": 0.0, "b": 1.0}
        sim = SimilarityMetrics.cosine_similarity(vec_a, vec_b)
        assert abs(sim) < 1e-6

    def test_cosine_no_common_keys(self):
        vec_a = {"a": 1.0}
        vec_b = {"b": 1.0}
        sim = SimilarityMetrics.cosine_similarity(vec_a, vec_b)
        assert sim == 0.0

    def test_cosine_empty_vectors(self):
        assert SimilarityMetrics.cosine_similarity({}, {}) == 0.0

    def test_cosine_zero_norm(self):
        vec_a = {"a": 0.0}
        vec_b = {"a": 1.0}
        assert SimilarityMetrics.cosine_similarity(vec_a, vec_b) == 0.0

    def test_pearson_perfect_positive(self):
        vec_a = {"x": 1.0, "y": 2.0, "z": 3.0}
        vec_b = {"x": 2.0, "y": 4.0, "z": 6.0}
        corr = SimilarityMetrics.pearson_correlation(vec_a, vec_b)
        assert abs(corr - 1.0) < 1e-6

    def test_pearson_perfect_negative(self):
        vec_a = {"x": 1.0, "y": 2.0, "z": 3.0}
        vec_b = {"x": 6.0, "y": 4.0, "z": 2.0}
        corr = SimilarityMetrics.pearson_correlation(vec_a, vec_b)
        assert abs(corr - (-1.0)) < 1e-6

    def test_pearson_insufficient_common(self):
        vec_a = {"x": 1.0}
        vec_b = {"x": 2.0}
        assert SimilarityMetrics.pearson_correlation(vec_a, vec_b) == 0.0

    def test_jaccard_identical_sets(self):
        s = {"a", "b", "c"}
        assert SimilarityMetrics.jaccard_similarity(s, s) == 1.0

    def test_jaccard_disjoint_sets(self):
        assert SimilarityMetrics.jaccard_similarity({"a"}, {"b"}) == 0.0

    def test_jaccard_partial_overlap(self):
        sim = SimilarityMetrics.jaccard_similarity({"a", "b", "c"}, {"b", "c", "d"})
        assert abs(sim - 0.5) < 1e-6

    def test_jaccard_empty_sets(self):
        assert SimilarityMetrics.jaccard_similarity(set(), set()) == 0.0


# ---------------------------------------------------------------------------
# CollaborativeFilter Tests
# ---------------------------------------------------------------------------

class TestCollaborativeFilter:
    """Tests for collaborative filtering recommender."""

    def test_user_based_fit(self, sample_interactions):
        cf = CollaborativeFilter(mode="user")
        cf.fit(sample_interactions)
        assert cf.is_fitted
        assert len(cf.user_item_matrix) == 4
        assert len(cf.item_user_matrix) == 5

    def test_item_based_fit(self, sample_interactions):
        cf = CollaborativeFilter(mode="item")
        cf.fit(sample_interactions)
        assert cf.is_fitted

    def test_method_chaining(self, sample_interactions):
        cf = CollaborativeFilter()
        result = cf.fit(sample_interactions)
        assert result is cf

    def test_predict_known_rating(self, sample_interactions):
        cf = CollaborativeFilter(mode="user")
        cf.fit(sample_interactions)
        rating = cf.predict("u1", "i1")
        assert rating == 5.0

    def test_predict_unknown_rating(self, sample_interactions):
        cf = CollaborativeFilter(mode="user", min_common_items=1)
        cf.fit(sample_interactions)
        rating = cf.predict("u1", "i4")
        assert isinstance(rating, float)

    def test_predict_not_fitted(self):
        cf = CollaborativeFilter()
        with pytest.raises(RuntimeError, match="fitted"):
            cf.predict("u1", "i1")

    def test_recommend_returns_list(self, sample_interactions):
        cf = CollaborativeFilter(mode="user", min_common_items=1)
        cf.fit(sample_interactions)
        recs = cf.recommend("u1", n=3)
        assert isinstance(recs, list)
        for rec in recs:
            assert "item_id" in rec
            assert "score" in rec
            assert "rank" in rec

    def test_recommend_excludes_seen(self, sample_interactions):
        cf = CollaborativeFilter(mode="user", min_common_items=1)
        cf.fit(sample_interactions)
        recs = cf.recommend("u1", n=5)
        seen = set(cf.user_item_matrix["u1"].keys())
        for rec in recs:
            assert rec["item_id"] not in seen

    def test_recommend_sorted_by_score(self, sample_interactions):
        cf = CollaborativeFilter(mode="user", min_common_items=1)
        cf.fit(sample_interactions)
        recs = cf.recommend("u1", n=5)
        scores = [r["score"] for r in recs]
        assert scores == sorted(scores, reverse=True)

    def test_recommend_not_fitted(self):
        cf = CollaborativeFilter()
        with pytest.raises(RuntimeError):
            cf.recommend("u1")

    def test_global_mean_calculation(self, sample_interactions):
        cf = CollaborativeFilter()
        cf.fit(sample_interactions)
        expected = sum(i["rating"] for i in sample_interactions) / len(sample_interactions)
        assert abs(cf.global_mean - expected) < 1e-6

    def test_user_means_calculation(self, sample_interactions):
        cf = CollaborativeFilter()
        cf.fit(sample_interactions)
        assert abs(cf.user_means["u1"] - 4.0) < 1e-6  # (5+4+3)/3

    def test_item_based_recommend(self, sample_interactions):
        cf = CollaborativeFilter(mode="item", min_common_items=1)
        cf.fit(sample_interactions)
        recs = cf.recommend("u1", n=2)
        assert isinstance(recs, list)

    def test_invalid_similarity_metric(self, sample_interactions):
        cf = CollaborativeFilter(similarity_metric="invalid")
        cf.fit(sample_interactions)
        with pytest.raises(ValueError, match="Unknown similarity"):
            cf.recommend("u1", n=1)

    def test_pearson_similarity_mode(self, sample_interactions):
        cf = CollaborativeFilter(similarity_metric="pearson", min_common_items=1)
        cf.fit(sample_interactions)
        recs = cf.recommend("u1", n=2)
        assert isinstance(recs, list)

    def test_jaccard_similarity_mode(self, sample_interactions):
        cf = CollaborativeFilter(similarity_metric="jaccard", min_common_items=1)
        cf.fit(sample_interactions)
        recs = cf.recommend("u1", n=2)
        assert isinstance(recs, list)


# ---------------------------------------------------------------------------
# ContentBasedFilter Tests
# ---------------------------------------------------------------------------

class TestContentBasedFilter:
    """Tests for content-based recommender."""

    def test_fit(self, sample_interactions, sample_item_features):
        cb = ContentBasedFilter()
        cb.fit(sample_interactions, sample_item_features)
        assert cb.is_fitted
        assert len(cb.user_profiles) == 4

    def test_method_chaining(self, sample_interactions, sample_item_features):
        cb = ContentBasedFilter()
        result = cb.fit(sample_interactions, sample_item_features)
        assert result is cb

    def test_recommend(self, sample_interactions, sample_item_features):
        cb = ContentBasedFilter()
        cb.fit(sample_interactions, sample_item_features)
        recs = cb.recommend("u1", n=3)
        assert isinstance(recs, list)
        assert len(recs) <= 3

    def test_recommend_excludes_seen(self, sample_interactions, sample_item_features):
        cb = ContentBasedFilter()
        cb.fit(sample_interactions, sample_item_features)
        recs = cb.recommend("u1", n=5)
        seen = set(cb.user_item_matrix["u1"].keys())
        for rec in recs:
            assert rec["item_id"] not in seen

    def test_recommend_not_fitted(self):
        cb = ContentBasedFilter()
        with pytest.raises(RuntimeError):
            cb.recommend("u1")

    def test_unknown_user(self, sample_interactions, sample_item_features):
        cb = ContentBasedFilter()
        cb.fit(sample_interactions, sample_item_features)
        recs = cb.recommend("unknown_user", n=3)
        assert recs == []

    def test_jaccard_similarity(self, sample_interactions, sample_item_features):
        cb = ContentBasedFilter(similarity_metric="jaccard")
        cb.fit(sample_interactions, sample_item_features)
        recs = cb.recommend("u1", n=3)
        assert isinstance(recs, list)


# ---------------------------------------------------------------------------
# HybridRecommender Tests
# ---------------------------------------------------------------------------

class TestHybridRecommender:
    """Tests for hybrid recommender."""

    def test_weighted_strategy(self, sample_interactions, sample_item_features):
        hr = HybridRecommender(strategy="weighted", cf_weight=0.6, cb_weight=0.4)
        hr.fit(sample_interactions, sample_item_features)
        assert hr.is_fitted
        recs = hr.recommend("u1", n=3)
        assert isinstance(recs, list)

    def test_switching_strategy(self, sample_interactions, sample_item_features):
        hr = HybridRecommender(strategy="switching")
        hr.fit(sample_interactions, sample_item_features)
        recs = hr.recommend("u1", n=3)
        assert isinstance(recs, list)

    def test_cascade_strategy(self, sample_interactions, sample_item_features):
        hr = HybridRecommender(strategy="cascade")
        hr.fit(sample_interactions, sample_item_features)
        recs = hr.recommend("u1", n=3)
        assert isinstance(recs, list)

    def test_invalid_weights(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            HybridRecommender(cf_weight=0.5, cb_weight=0.3)

    def test_invalid_strategy(self, sample_interactions):
        hr = HybridRecommender(strategy="invalid")
        hr.fit(sample_interactions)
        with pytest.raises(ValueError, match="Unknown strategy"):
            hr.recommend("u1")

    def test_not_fitted(self):
        hr = HybridRecommender()
        with pytest.raises(RuntimeError):
            hr.recommend("u1")

    def test_cf_only(self, sample_interactions):
        """Test hybrid without item features (CF only)."""
        hr = HybridRecommender()
        hr.fit(sample_interactions)
        assert hr.is_fitted
        recs = hr.recommend("u1", n=2)
        assert isinstance(recs, list)


# ---------------------------------------------------------------------------
# RecommenderEvaluator Tests
# ---------------------------------------------------------------------------

class TestRecommenderEvaluator:
    """Tests for evaluation metrics."""

    def setup_method(self):
        self.evaluator = RecommenderEvaluator()

    def test_precision_at_k_perfect(self):
        recommended = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert self.evaluator.precision_at_k(recommended, relevant, 3) == 1.0

    def test_precision_at_k_none(self):
        recommended = ["x", "y", "z"]
        relevant = {"a", "b", "c"}
        assert self.evaluator.precision_at_k(recommended, relevant, 3) == 0.0

    def test_precision_at_k_partial(self):
        recommended = ["a", "x", "b"]
        relevant = {"a", "b", "c"}
        p = self.evaluator.precision_at_k(recommended, relevant, 3)
        assert abs(p - 2 / 3) < 1e-6

    def test_precision_at_k_zero(self):
        assert self.evaluator.precision_at_k([], set(), 0) == 0.0

    def test_recall_at_k_perfect(self):
        recommended = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert self.evaluator.recall_at_k(recommended, relevant, 3) == 1.0

    def test_recall_at_k_partial(self):
        recommended = ["a", "x"]
        relevant = {"a", "b"}
        r = self.evaluator.recall_at_k(recommended, relevant, 2)
        assert abs(r - 0.5) < 1e-6

    def test_recall_at_k_empty_relevant(self):
        assert self.evaluator.recall_at_k(["a"], set(), 1) == 0.0

    def test_ndcg_at_k_perfect(self):
        recommended = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        ndcg = self.evaluator.ndcg_at_k(recommended, relevant, 3)
        assert abs(ndcg - 1.0) < 1e-6

    def test_ndcg_at_k_zero(self):
        recommended = ["x", "y"]
        relevant = {"a", "b"}
        ndcg = self.evaluator.ndcg_at_k(recommended, relevant, 2)
        assert ndcg == 0.0

    def test_ndcg_at_k_empty_relevant(self):
        assert self.evaluator.ndcg_at_k(["a"], set(), 1) == 0.0

    def test_map_at_k_perfect(self):
        recommended = ["a", "b"]
        relevant = {"a", "b"}
        score = self.evaluator.mean_average_precision(recommended, relevant, 2)
        assert abs(score - 1.0) < 1e-6

    def test_map_at_k_zero(self):
        recommended = ["x", "y"]
        relevant = {"a", "b"}
        score = self.evaluator.mean_average_precision(recommended, relevant, 2)
        assert score == 0.0

    def test_evaluate_aggregated(self):
        recs = {"u1": ["a", "b"], "u2": ["c", "d"]}
        truth = {"u1": {"a", "c"}, "u2": {"c", "e"}}
        result = self.evaluator.evaluate(recs, truth, k=2)
        assert "precision_at_k" in result
        assert "recall_at_k" in result
        assert "ndcg_at_k" in result
        assert "map_at_k" in result
        assert "num_users_evaluated" in result
        assert result["num_users_evaluated"] == 2

    def test_evaluate_no_ground_truth(self):
        recs = {"u1": ["a", "b"]}
        truth = {}
        result = self.evaluator.evaluate(recs, truth, k=2)
        assert result["num_users_evaluated"] == 0


# ---------------------------------------------------------------------------
# Integration & Structure Tests
# ---------------------------------------------------------------------------

class TestProjectStructure:
    """Verify project structure and configuration."""

    def test_src_directory_exists(self):
        src = Path(__file__).parent.parent / "src"
        assert src.exists()

    def test_requirements_file_exists(self):
        req = Path(__file__).parent.parent / "requirements.txt"
        assert req.exists()

    def test_readme_exists_and_substantial(self):
        readme = Path(__file__).parent.parent / "README.md"
        assert readme.exists()
        content = readme.read_text(encoding="utf-8")
        assert len(content) > 500

    def test_license_exists(self):
        license_file = Path(__file__).parent.parent / "LICENSE"
        assert license_file.exists()


class TestIntegration:
    """End-to-end integration tests."""

    def test_full_pipeline(self, sample_interactions, sample_item_features):
        """Test complete recommendation pipeline."""
        # Train
        hr = HybridRecommender(strategy="weighted")
        hr.fit(sample_interactions, sample_item_features)

        # Recommend
        recs = hr.recommend("u1", n=3)
        assert len(recs) > 0
        assert all("item_id" in r and "score" in r for r in recs)

        # Evaluate
        evaluator = RecommenderEvaluator()
        all_recs = {}
        ground_truth = {}
        for uid in ["u1", "u2", "u3", "u4"]:
            user_recs = hr.recommend(uid, n=5)
            all_recs[uid] = [r["item_id"] for r in user_recs]
            ground_truth[uid] = set(hr.cf.user_item_matrix.get(uid, {}).keys())

        metrics = evaluator.evaluate(all_recs, ground_truth, k=5)
        assert metrics["num_users_evaluated"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
