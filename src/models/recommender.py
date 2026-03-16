"""
Recommender Systems Engine Module.

Implements collaborative filtering, content-based filtering, and hybrid
recommendation algorithms using matrix factorization and similarity metrics.
"""

from typing import List, Dict, Optional, Any, Tuple
import math
import random
from collections import defaultdict


class SimilarityMetrics:
    """Collection of similarity computation methods."""

    @staticmethod
    def cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
        """
        Compute cosine similarity between two sparse vectors.

        Args:
            vec_a: First sparse vector as {dimension: value}
            vec_b: Second sparse vector as {dimension: value}

        Returns:
            Cosine similarity in range [-1, 1]
        """
        common_keys = set(vec_a.keys()) & set(vec_b.keys())
        if not common_keys:
            return 0.0

        dot_product = sum(vec_a[k] * vec_b[k] for k in common_keys)
        norm_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
        norm_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    @staticmethod
    def pearson_correlation(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
        """
        Compute Pearson correlation coefficient between two sparse vectors.

        Args:
            vec_a: First sparse vector as {dimension: value}
            vec_b: Second sparse vector as {dimension: value}

        Returns:
            Pearson correlation in range [-1, 1]
        """
        common_keys = set(vec_a.keys()) & set(vec_b.keys())
        n = len(common_keys)
        if n < 2:
            return 0.0

        vals_a = [vec_a[k] for k in common_keys]
        vals_b = [vec_b[k] for k in common_keys]

        mean_a = sum(vals_a) / n
        mean_b = sum(vals_b) / n

        numerator = sum((a - mean_a) * (b - mean_b) for a, b in zip(vals_a, vals_b))
        denom_a = math.sqrt(sum((a - mean_a) ** 2 for a in vals_a))
        denom_b = math.sqrt(sum((b - mean_b) ** 2 for b in vals_b))

        if denom_a == 0 or denom_b == 0:
            return 0.0

        return numerator / (denom_a * denom_b)

    @staticmethod
    def jaccard_similarity(set_a: set, set_b: set) -> float:
        """
        Compute Jaccard similarity between two sets.

        Args:
            set_a: First set of items
            set_b: Second set of items

        Returns:
            Jaccard similarity in range [0, 1]
        """
        if not set_a and not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0


class CollaborativeFilter:
    """
    User-based and item-based collaborative filtering.

    Uses nearest-neighbor approach with configurable similarity metrics
    to generate recommendations from user-item interaction data.
    """

    def __init__(
        self,
        mode: str = "user",
        similarity_metric: str = "cosine",
        n_neighbors: int = 20,
        min_common_items: int = 2,
    ):
        """
        Initialize collaborative filter.

        Args:
            mode: 'user' for user-based or 'item' for item-based CF
            similarity_metric: 'cosine', 'pearson', or 'jaccard'
            n_neighbors: Number of neighbors to consider
            min_common_items: Minimum shared items for valid similarity
        """
        self.mode = mode
        self.similarity_metric = similarity_metric
        self.n_neighbors = n_neighbors
        self.min_common_items = min_common_items

        self.user_item_matrix: Dict[str, Dict[str, float]] = {}
        self.item_user_matrix: Dict[str, Dict[str, float]] = {}
        self.user_means: Dict[str, float] = {}
        self.global_mean: float = 0.0
        self.is_fitted: bool = False

        self._metrics = SimilarityMetrics()

    def fit(self, interactions: List[Dict[str, Any]]) -> "CollaborativeFilter":
        """
        Build the user-item matrix from interaction data.

        Args:
            interactions: List of dicts with 'user_id', 'item_id', 'rating'

        Returns:
            self for method chaining
        """
        self.user_item_matrix.clear()
        self.item_user_matrix.clear()

        all_ratings = []
        for interaction in interactions:
            user_id = str(interaction["user_id"])
            item_id = str(interaction["item_id"])
            rating = float(interaction.get("rating", 1.0))

            self.user_item_matrix.setdefault(user_id, {})[item_id] = rating
            self.item_user_matrix.setdefault(item_id, {})[user_id] = rating
            all_ratings.append(rating)

        self.global_mean = sum(all_ratings) / len(all_ratings) if all_ratings else 0.0

        for user_id, ratings in self.user_item_matrix.items():
            vals = list(ratings.values())
            self.user_means[user_id] = sum(vals) / len(vals) if vals else 0.0

        self.is_fitted = True
        return self

    def _compute_similarity(
        self, vec_a: Dict[str, float], vec_b: Dict[str, float]
    ) -> float:
        """Compute similarity based on configured metric."""
        common = set(vec_a.keys()) & set(vec_b.keys())
        if len(common) < self.min_common_items:
            return 0.0

        if self.similarity_metric == "cosine":
            return self._metrics.cosine_similarity(vec_a, vec_b)
        elif self.similarity_metric == "pearson":
            return self._metrics.pearson_correlation(vec_a, vec_b)
        elif self.similarity_metric == "jaccard":
            return self._metrics.jaccard_similarity(set(vec_a.keys()), set(vec_b.keys()))
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")

    def _find_neighbors(
        self, target_id: str, matrix: Dict[str, Dict[str, float]]
    ) -> List[Tuple[str, float]]:
        """Find k nearest neighbors for a given entity."""
        similarities = []
        target_vec = matrix.get(target_id, {})
        if not target_vec:
            return []

        for other_id, other_vec in matrix.items():
            if other_id == target_id:
                continue
            sim = self._compute_similarity(target_vec, other_vec)
            if sim > 0:
                similarities.append((other_id, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[: self.n_neighbors]

    def predict(self, user_id: str, item_id: str) -> float:
        """
        Predict rating for a user-item pair.

        Args:
            user_id: Target user ID
            item_id: Target item ID

        Returns:
            Predicted rating value
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")

        user_id = str(user_id)
        item_id = str(item_id)

        if user_id in self.user_item_matrix and item_id in self.user_item_matrix[user_id]:
            return self.user_item_matrix[user_id][item_id]

        if self.mode == "user":
            return self._predict_user_based(user_id, item_id)
        else:
            return self._predict_item_based(user_id, item_id)

    def _predict_user_based(self, user_id: str, item_id: str) -> float:
        """User-based collaborative filtering prediction."""
        neighbors = self._find_neighbors(user_id, self.user_item_matrix)

        weighted_sum = 0.0
        sim_sum = 0.0

        for neighbor_id, similarity in neighbors:
            neighbor_ratings = self.user_item_matrix.get(neighbor_id, {})
            if item_id in neighbor_ratings:
                neighbor_mean = self.user_means.get(neighbor_id, self.global_mean)
                deviation = neighbor_ratings[item_id] - neighbor_mean
                weighted_sum += similarity * deviation
                sim_sum += abs(similarity)

        user_mean = self.user_means.get(user_id, self.global_mean)

        if sim_sum == 0:
            return user_mean

        return user_mean + (weighted_sum / sim_sum)

    def _predict_item_based(self, user_id: str, item_id: str) -> float:
        """Item-based collaborative filtering prediction."""
        neighbors = self._find_neighbors(item_id, self.item_user_matrix)

        weighted_sum = 0.0
        sim_sum = 0.0

        user_ratings = self.user_item_matrix.get(user_id, {})
        for neighbor_item_id, similarity in neighbors:
            if neighbor_item_id in user_ratings:
                weighted_sum += similarity * user_ratings[neighbor_item_id]
                sim_sum += abs(similarity)

        if sim_sum == 0:
            return self.user_means.get(user_id, self.global_mean)

        return weighted_sum / sim_sum

    def recommend(self, user_id: str, n: int = 10) -> List[Dict[str, Any]]:
        """
        Generate top-N recommendations for a user.

        Args:
            user_id: Target user ID
            n: Number of recommendations

        Returns:
            List of recommended items with predicted scores
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before generating recommendations")

        user_id = str(user_id)
        seen_items = set(self.user_item_matrix.get(user_id, {}).keys())
        all_items = set(self.item_user_matrix.keys())
        candidate_items = all_items - seen_items

        scored_items = []
        for item_id in candidate_items:
            score = self.predict(user_id, item_id)
            scored_items.append({"item_id": item_id, "score": round(score, 4)})

        scored_items.sort(key=lambda x: x["score"], reverse=True)

        results = scored_items[:n]
        for rank, item in enumerate(results, 1):
            item["rank"] = rank

        return results


class ContentBasedFilter:
    """
    Content-based recommendation using item feature similarity.

    Builds user profiles from rated item features and recommends
    items with similar feature profiles.
    """

    def __init__(self, similarity_metric: str = "cosine"):
        """
        Initialize content-based filter.

        Args:
            similarity_metric: 'cosine' or 'jaccard'
        """
        self.similarity_metric = similarity_metric
        self.item_features: Dict[str, Dict[str, float]] = {}
        self.user_profiles: Dict[str, Dict[str, float]] = {}
        self.user_item_matrix: Dict[str, Dict[str, float]] = {}
        self.is_fitted: bool = False
        self._metrics = SimilarityMetrics()

    def fit(
        self,
        interactions: List[Dict[str, Any]],
        item_features: Dict[str, Dict[str, float]],
    ) -> "ContentBasedFilter":
        """
        Build user profiles from interactions and item features.

        Args:
            interactions: List of dicts with 'user_id', 'item_id', 'rating'
            item_features: Dict mapping item_id to feature vectors

        Returns:
            self for method chaining
        """
        self.item_features = {str(k): v for k, v in item_features.items()}
        self.user_item_matrix.clear()
        self.user_profiles.clear()

        for interaction in interactions:
            user_id = str(interaction["user_id"])
            item_id = str(interaction["item_id"])
            rating = float(interaction.get("rating", 1.0))
            self.user_item_matrix.setdefault(user_id, {})[item_id] = rating

        for user_id, ratings in self.user_item_matrix.items():
            profile: Dict[str, float] = defaultdict(float)
            total_weight = 0.0

            for item_id, rating in ratings.items():
                features = self.item_features.get(item_id, {})
                for feature, value in features.items():
                    profile[feature] += rating * value
                total_weight += rating

            if total_weight > 0:
                self.user_profiles[user_id] = {
                    k: v / total_weight for k, v in profile.items()
                }

        self.is_fitted = True
        return self

    def recommend(self, user_id: str, n: int = 10) -> List[Dict[str, Any]]:
        """
        Generate content-based recommendations.

        Args:
            user_id: Target user ID
            n: Number of recommendations

        Returns:
            List of recommended items with similarity scores
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before generating recommendations")

        user_id = str(user_id)
        user_profile = self.user_profiles.get(user_id, {})
        if not user_profile:
            return []

        seen_items = set(self.user_item_matrix.get(user_id, {}).keys())
        candidates = []

        for item_id, features in self.item_features.items():
            if item_id in seen_items:
                continue

            if self.similarity_metric == "cosine":
                score = self._metrics.cosine_similarity(user_profile, features)
            else:
                score = self._metrics.jaccard_similarity(
                    set(user_profile.keys()), set(features.keys())
                )
            candidates.append({"item_id": item_id, "score": round(score, 4)})

        candidates.sort(key=lambda x: x["score"], reverse=True)

        results = candidates[:n]
        for rank, item in enumerate(results, 1):
            item["rank"] = rank

        return results


class HybridRecommender:
    """
    Hybrid recommender combining collaborative and content-based approaches.

    Supports weighted combination, switching, and cascade strategies
    for merging recommendations from multiple sub-models.
    """

    def __init__(
        self,
        strategy: str = "weighted",
        cf_weight: float = 0.6,
        cb_weight: float = 0.4,
        cf_mode: str = "user",
        similarity_metric: str = "cosine",
    ):
        """
        Initialize hybrid recommender.

        Args:
            strategy: 'weighted', 'switching', or 'cascade'
            cf_weight: Weight for collaborative filtering scores
            cb_weight: Weight for content-based scores
            cf_mode: 'user' or 'item' for collaborative filter
            similarity_metric: Similarity metric for both sub-models
        """
        if abs(cf_weight + cb_weight - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")

        self.strategy = strategy
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight

        self.cf = CollaborativeFilter(
            mode=cf_mode, similarity_metric=similarity_metric
        )
        self.cb = ContentBasedFilter(similarity_metric=similarity_metric)
        self.is_fitted: bool = False

    def fit(
        self,
        interactions: List[Dict[str, Any]],
        item_features: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> "HybridRecommender":
        """
        Train both sub-models.

        Args:
            interactions: User-item interaction data
            item_features: Item feature vectors (required for content-based)

        Returns:
            self for method chaining
        """
        self.cf.fit(interactions)

        if item_features:
            self.cb.fit(interactions, item_features)

        self.is_fitted = True
        return self

    def recommend(self, user_id: str, n: int = 10) -> List[Dict[str, Any]]:
        """
        Generate hybrid recommendations.

        Args:
            user_id: Target user ID
            n: Number of recommendations

        Returns:
            Merged and ranked recommendation list
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before generating recommendations")

        if self.strategy == "weighted":
            return self._weighted_recommend(user_id, n)
        elif self.strategy == "switching":
            return self._switching_recommend(user_id, n)
        elif self.strategy == "cascade":
            return self._cascade_recommend(user_id, n)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _weighted_recommend(self, user_id: str, n: int) -> List[Dict[str, Any]]:
        """Combine scores from both models using weighted average."""
        cf_recs = self.cf.recommend(user_id, n * 2)
        cb_recs = self.cb.recommend(user_id, n * 2) if self.cb.is_fitted else []

        score_map: Dict[str, float] = {}

        for rec in cf_recs:
            item_id = rec["item_id"]
            score_map[item_id] = score_map.get(item_id, 0.0) + self.cf_weight * rec["score"]

        for rec in cb_recs:
            item_id = rec["item_id"]
            score_map[item_id] = score_map.get(item_id, 0.0) + self.cb_weight * rec["score"]

        merged = [
            {"item_id": item_id, "score": round(score, 4)}
            for item_id, score in score_map.items()
        ]
        merged.sort(key=lambda x: x["score"], reverse=True)

        results = merged[:n]
        for rank, item in enumerate(results, 1):
            item["rank"] = rank

        return results

    def _switching_recommend(self, user_id: str, n: int) -> List[Dict[str, Any]]:
        """Switch to content-based when user has few interactions."""
        user_interactions = len(self.cf.user_item_matrix.get(str(user_id), {}))
        cold_start_threshold = 5

        if user_interactions < cold_start_threshold and self.cb.is_fitted:
            return self.cb.recommend(user_id, n)
        return self.cf.recommend(user_id, n)

    def _cascade_recommend(self, user_id: str, n: int) -> List[Dict[str, Any]]:
        """Use CF as primary, re-rank top results with content similarity."""
        cf_recs = self.cf.recommend(user_id, n * 3)

        if not self.cb.is_fitted:
            return cf_recs[:n]

        user_profile = self.cb.user_profiles.get(str(user_id), {})
        if not user_profile:
            return cf_recs[:n]

        for rec in cf_recs:
            item_features = self.cb.item_features.get(rec["item_id"], {})
            content_score = self._cb_metrics_cosine(user_profile, item_features)
            rec["score"] = round(
                self.cf_weight * rec["score"] + self.cb_weight * content_score, 4
            )

        cf_recs.sort(key=lambda x: x["score"], reverse=True)
        results = cf_recs[:n]
        for rank, item in enumerate(results, 1):
            item["rank"] = rank
        return results

    def _cb_metrics_cosine(
        self, vec_a: Dict[str, float], vec_b: Dict[str, float]
    ) -> float:
        """Helper for cosine similarity in cascade mode."""
        return SimilarityMetrics.cosine_similarity(vec_a, vec_b)


class RecommenderEvaluator:
    """
    Evaluation framework for recommendation models.

    Computes standard information retrieval metrics including
    Precision@K, Recall@K, NDCG@K, and MAP@K.
    """

    @staticmethod
    def precision_at_k(recommended: List[str], relevant: set, k: int) -> float:
        """
        Compute Precision@K.

        Args:
            recommended: Ordered list of recommended item IDs
            relevant: Set of relevant (ground truth) item IDs
            k: Cutoff position

        Returns:
            Precision@K score
        """
        if k <= 0:
            return 0.0
        top_k = recommended[:k]
        hits = sum(1 for item in top_k if item in relevant)
        return hits / k

    @staticmethod
    def recall_at_k(recommended: List[str], relevant: set, k: int) -> float:
        """
        Compute Recall@K.

        Args:
            recommended: Ordered list of recommended item IDs
            relevant: Set of relevant (ground truth) item IDs
            k: Cutoff position

        Returns:
            Recall@K score
        """
        if not relevant or k <= 0:
            return 0.0
        top_k = recommended[:k]
        hits = sum(1 for item in top_k if item in relevant)
        return hits / len(relevant)

    @staticmethod
    def ndcg_at_k(recommended: List[str], relevant: set, k: int) -> float:
        """
        Compute Normalized Discounted Cumulative Gain at K.

        Args:
            recommended: Ordered list of recommended item IDs
            relevant: Set of relevant (ground truth) item IDs
            k: Cutoff position

        Returns:
            NDCG@K score
        """
        if k <= 0 or not relevant:
            return 0.0

        top_k = recommended[:k]

        dcg = 0.0
        for i, item in enumerate(top_k):
            if item in relevant:
                dcg += 1.0 / math.log2(i + 2)

        ideal_hits = min(len(relevant), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def mean_average_precision(
        recommended: List[str], relevant: set, k: int
    ) -> float:
        """
        Compute Mean Average Precision at K.

        Args:
            recommended: Ordered list of recommended item IDs
            relevant: Set of relevant (ground truth) item IDs
            k: Cutoff position

        Returns:
            MAP@K score
        """
        if not relevant or k <= 0:
            return 0.0

        top_k = recommended[:k]
        score = 0.0
        hits = 0

        for i, item in enumerate(top_k):
            if item in relevant:
                hits += 1
                score += hits / (i + 1)

        return score / min(len(relevant), k)

    def evaluate(
        self,
        recommendations: Dict[str, List[str]],
        ground_truth: Dict[str, set],
        k: int = 10,
    ) -> Dict[str, float]:
        """
        Evaluate recommendations across all users.

        Args:
            recommendations: Dict mapping user_id to recommended item lists
            ground_truth: Dict mapping user_id to relevant item sets
            k: Cutoff position

        Returns:
            Dict of averaged metric scores
        """
        precisions = []
        recalls = []
        ndcgs = []
        maps = []

        for user_id, recs in recommendations.items():
            relevant = ground_truth.get(user_id, set())
            if not relevant:
                continue

            precisions.append(self.precision_at_k(recs, relevant, k))
            recalls.append(self.recall_at_k(recs, relevant, k))
            ndcgs.append(self.ndcg_at_k(recs, relevant, k))
            maps.append(self.mean_average_precision(recs, relevant, k))

        n = len(precisions) if precisions else 1

        return {
            "precision_at_k": round(sum(precisions) / n, 4) if precisions else 0.0,
            "recall_at_k": round(sum(recalls) / n, 4) if recalls else 0.0,
            "ndcg_at_k": round(sum(ndcgs) / n, 4) if ndcgs else 0.0,
            "map_at_k": round(sum(maps) / n, 4) if maps else 0.0,
            "num_users_evaluated": len(precisions),
        }
