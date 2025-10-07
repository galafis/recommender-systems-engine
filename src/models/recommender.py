"""
Recommender Systems Engine

Collaborative filtering and content-based recommendation system.

Author: Gabriel Demetrios Lafis
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from typing import List, Dict, Tuple
from loguru import logger


class CollaborativeFilteringRecommender:
    """
    Collaborative filtering recommender using matrix factorization.
    """
    
    def __init__(
        self,
        n_factors: int = 50,
        algorithm: str = 'svd'
    ):
        """
        Initialize recommender.
        
        Args:
            n_factors: Number of latent factors
            algorithm: Algorithm to use ('svd', 'als')
        """
        self.n_factors = n_factors
        self.algorithm = algorithm
        self.model = None
        self.user_factors = None
        self.item_factors = None
        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_user_mapping = {}
        self.reverse_item_mapping = {}
        
        logger.info(f"Initialized {algorithm.upper()} recommender with {n_factors} factors")
    
    def fit(
        self,
        user_item_matrix: np.ndarray,
        user_ids: List = None,
        item_ids: List = None
    ):
        """
        Fit recommender model.
        
        Args:
            user_item_matrix: User-item interaction matrix
            user_ids: List of user IDs
            item_ids: List of item IDs
        """
        logger.info("Training recommender model...")
        
        # Create mappings
        if user_ids is not None:
            self.user_mapping = {uid: idx for idx, uid in enumerate(user_ids)}
            self.reverse_user_mapping = {idx: uid for uid, idx in self.user_mapping.items()}
        
        if item_ids is not None:
            self.item_mapping = {iid: idx for idx, iid in enumerate(item_ids)}
            self.reverse_item_mapping = {idx: iid for iid, idx in self.item_mapping.items()}
        
        # Train model
        if self.algorithm == 'svd':
            self._fit_svd(user_item_matrix)
        elif self.algorithm == 'als':
            self._fit_als(user_item_matrix)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        logger.success("Model trained successfully")
    
    def _fit_svd(self, user_item_matrix: np.ndarray):
        """Fit using SVD."""
        self.model = TruncatedSVD(n_components=self.n_factors, random_state=42)
        self.user_factors = self.model.fit_transform(user_item_matrix)
        self.item_factors = self.model.components_.T
    
    def _fit_als(self, user_item_matrix: np.ndarray, n_iterations: int = 10):
        """Fit using Alternating Least Squares."""
        n_users, n_items = user_item_matrix.shape
        
        # Initialize factors randomly
        self.user_factors = np.random.rand(n_users, self.n_factors)
        self.item_factors = np.random.rand(n_items, self.n_factors)
        
        # ALS iterations
        for iteration in range(n_iterations):
            # Update user factors
            for u in range(n_users):
                items = user_item_matrix[u, :].nonzero()[0]
                if len(items) > 0:
                    A = self.item_factors[items]
                    b = user_item_matrix[u, items]
                    self.user_factors[u] = np.linalg.lstsq(A, b, rcond=None)[0]
            
            # Update item factors
            for i in range(n_items):
                users = user_item_matrix[:, i].nonzero()[0]
                if len(users) > 0:
                    A = self.user_factors[users]
                    b = user_item_matrix[users, i]
                    self.item_factors[i] = np.linalg.lstsq(A, b, rcond=None)[0]
    
    def recommend(
        self,
        user_id: int,
        n_recommendations: int = 10,
        exclude_known: bool = True,
        known_items: List[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Generate recommendations for a user.
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations
            exclude_known: Whether to exclude known items
            known_items: List of known item IDs
            
        Returns:
            List of (item_id, score) tuples
        """
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Model not trained yet")
        
        # Get user index
        user_idx = self.user_mapping.get(user_id, user_id)
        
        # Calculate scores
        user_vector = self.user_factors[user_idx]
        scores = np.dot(self.item_factors, user_vector)
        
        # Exclude known items
        if exclude_known and known_items is not None:
            known_indices = [self.item_mapping.get(iid, iid) for iid in known_items]
            scores[known_indices] = -np.inf
        
        # Get top-N recommendations
        top_indices = np.argsort(scores)[::-1][:n_recommendations]
        
        recommendations = [
            (self.reverse_item_mapping.get(idx, idx), scores[idx])
            for idx in top_indices
        ]
        
        return recommendations
    
    def similar_items(
        self,
        item_id: int,
        n_similar: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Find similar items.
        
        Args:
            item_id: Item ID
            n_similar: Number of similar items
            
        Returns:
            List of (item_id, similarity) tuples
        """
        if self.item_factors is None:
            raise ValueError("Model not trained yet")
        
        # Get item index
        item_idx = self.item_mapping.get(item_id, item_id)
        
        # Calculate similarities
        item_vector = self.item_factors[item_idx].reshape(1, -1)
        similarities = cosine_similarity(item_vector, self.item_factors)[0]
        
        # Get top-N similar items (excluding itself)
        similarities[item_idx] = -1
        top_indices = np.argsort(similarities)[::-1][:n_similar]
        
        similar_items = [
            (self.reverse_item_mapping.get(idx, idx), similarities[idx])
            for idx in top_indices
        ]
        
        return similar_items
    
    def evaluate(
        self,
        test_user_item_matrix: np.ndarray,
        k: int = 10
    ) -> Dict[str, float]:
        """
        Evaluate recommender performance.
        
        Args:
            test_user_item_matrix: Test user-item matrix
            k: Number of recommendations for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        n_users = test_user_item_matrix.shape[0]
        
        precisions = []
        recalls = []
        
        for user_idx in range(n_users):
            # Get ground truth
            true_items = test_user_item_matrix[user_idx].nonzero()[0]
            
            if len(true_items) == 0:
                continue
            
            # Get recommendations
            user_id = self.reverse_user_mapping.get(user_idx, user_idx)
            recommendations = self.recommend(user_id, n_recommendations=k)
            recommended_items = [item_id for item_id, _ in recommendations]
            
            # Calculate precision and recall
            hits = len(set(recommended_items) & set(true_items))
            precision = hits / k if k > 0 else 0
            recall = hits / len(true_items) if len(true_items) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
        
        metrics = {
            f'precision@{k}': np.mean(precisions),
            f'recall@{k}': np.mean(recalls)
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        
        return metrics


if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)
    
    # Create synthetic user-item matrix
    n_users = 100
    n_items = 50
    density = 0.1
    
    user_item_matrix = np.random.rand(n_users, n_items)
    user_item_matrix[user_item_matrix > density] = 0
    user_item_matrix[user_item_matrix > 0] = np.random.randint(1, 6, size=np.sum(user_item_matrix > 0))
    
    # Train recommender
    recommender = CollaborativeFilteringRecommender(n_factors=10, algorithm='svd')
    recommender.fit(user_item_matrix)
    
    # Get recommendations
    user_id = 0
    recommendations = recommender.recommend(user_id, n_recommendations=5)
    
    print(f"\nTop 5 recommendations for user {user_id}:")
    for item_id, score in recommendations:
        print(f"  Item {item_id}: {score:.4f}")
    
    # Find similar items
    item_id = 0
    similar = recommender.similar_items(item_id, n_similar=5)
    
    print(f"\nTop 5 similar items to item {item_id}:")
    for sim_item_id, similarity in similar:
        print(f"  Item {sim_item_id}: {similarity:.4f}")
