"""
Learning module for MX Clipping.

This module provides adaptive learning capabilities to improve clip detection
and generation based on user feedback and preferences.
"""

import os
import json
import logging
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
import sqlite3
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class ClipFeedbackTracker:
    """
    Tracks user feedback on clips and adapts trigger sensitivity based on patterns.
    
    Features:
    - Record whether clips are kept or discarded
    - Analyze patterns in successful clips
    - Adjust trigger parameters based on learning
    """
    
    def __init__(self, db_path: str = "feedback.db"):
        """
        Initialize the feedback tracker.
        
        Args:
            db_path: Path to the SQLite database for storing feedback
        """
        self.db_path = db_path
        self._initialize_db()
        
        # Cache for trigger adjustments
        self.trigger_adjustments = defaultdict(dict)
        self.last_analysis_time = 0
        self.analysis_interval = 3600  # Analyze once per hour
    
    def _initialize_db(self) -> None:
        """Initialize the SQLite database for storing feedback."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS clip_feedback (
            clip_id TEXT PRIMARY KEY,
            user_id TEXT,
            reason TEXT,
            is_kept INTEGER,
            metadata TEXT,
            timestamp TEXT
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS trigger_adjustments (
            user_id TEXT,
            trigger_type TEXT,
            adjustment_factor REAL,
            timestamp TEXT,
            PRIMARY KEY (user_id, trigger_type)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def track_clip_feedback(self, clip_id: str, user_id: str, reason: str, 
                           is_kept: bool, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Record feedback for a clip.
        
        Args:
            clip_id: Unique identifier for the clip
            user_id: User who provided the feedback
            reason: Reason the clip was created
            is_kept: Whether the clip was kept (True) or discarded (False)
            metadata: Additional metadata about the clip
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        metadata_json = json.dumps(metadata or {})
        timestamp = datetime.now().isoformat()
        
        cursor.execute(
            "INSERT OR REPLACE INTO clip_feedback VALUES (?, ?, ?, ?, ?, ?)",
            (clip_id, user_id, reason, int(is_kept), metadata_json, timestamp)
        )
        
        conn.commit()
        conn.close()
        
        # Check if we should analyze patterns
        current_time = time.time()
        if current_time - self.last_analysis_time > self.analysis_interval:
            self.analyze_feedback_patterns()
            self.last_analysis_time = current_time
    
    def analyze_feedback_patterns(self) -> Dict[str, Dict[str, float]]:
        """
        Analyze patterns in clip feedback to determine what works well.
        
        Returns:
            Dictionary mapping user_ids to trigger types and their adjustment factors
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all feedback grouped by user_id and reason
        cursor.execute('''
        SELECT user_id, reason, 
               SUM(is_kept) as kept_count,
               COUNT(*) as total_count
        FROM clip_feedback
        GROUP BY user_id, reason
        ''')
        
        results = cursor.fetchall()
        conn.close()
        
        # Calculate adjustment factors
        adjustments = defaultdict(dict)
        
        for user_id, reason, kept_count, total_count in results:
            if total_count < 5:  # Require minimum sample size
                continue
                
            success_rate = kept_count / total_count
            
            # Calculate adjustment factor (higher success = lower threshold)
            # Base adjustment on deviation from expected 50% success rate
            adjustment_factor = 1.0
            if success_rate > 0.7:
                # Very successful - make it more sensitive
                adjustment_factor = 0.8
            elif success_rate > 0.5:
                # Moderately successful - make it slightly more sensitive
                adjustment_factor = 0.9
            elif success_rate < 0.3:
                # Very unsuccessful - make it less sensitive
                adjustment_factor = 1.2
            elif success_rate < 0.5:
                # Moderately unsuccessful - make it slightly less sensitive
                adjustment_factor = 1.1
            
            # Extract the trigger type from the reason
            if ":" in reason:
                trigger_type = reason.split(":")[0]
            else:
                trigger_type = reason
                
            adjustments[user_id][trigger_type] = adjustment_factor
            
            # Store adjustment in database
            self._store_adjustment(user_id, trigger_type, adjustment_factor)
        
        # Update cache
        for user_id, user_adjustments in adjustments.items():
            self.trigger_adjustments[user_id].update(user_adjustments)
        
        return dict(adjustments)
    
    def _store_adjustment(self, user_id: str, trigger_type: str, adjustment_factor: float) -> None:
        """Store trigger adjustment in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        cursor.execute(
            "INSERT OR REPLACE INTO trigger_adjustments VALUES (?, ?, ?, ?)",
            (user_id, trigger_type, adjustment_factor, timestamp)
        )
        
        conn.commit()
        conn.close()
    
    def get_trigger_adjustment(self, user_id: str, trigger_type: str) -> float:
        """
        Get the adjustment factor for a specific trigger type and user.
        
        Args:
            user_id: User ID
            trigger_type: Type of trigger (e.g., 'keyword', 'emotion')
            
        Returns:
            Adjustment factor (multiply thresholds by this value)
        """
        # Check cache first
        if user_id in self.trigger_adjustments and trigger_type in self.trigger_adjustments[user_id]:
            return self.trigger_adjustments[user_id][trigger_type]
        
        # If not in cache, check database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT adjustment_factor FROM trigger_adjustments WHERE user_id = ? AND trigger_type = ?",
            (user_id, trigger_type)
        )
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            # Update cache and return
            self.trigger_adjustments[user_id][trigger_type] = result[0]
            return result[0]
        
        # Default to no adjustment
        return 1.0


class UserPreferenceModel:
    """
    Manages user preferences for clips based on historical feedback.
    
    Features:
    - Build user preference profiles based on clip history
    - Track preferred content types
    - Rank clip suggestions based on preferences
    """
    
    def __init__(self, db_path: str = "preferences.db"):
        """
        Initialize the user preference model.
        
        Args:
            db_path: Path to the SQLite database for storing preferences
        """
        self.db_path = db_path
        self._initialize_db()
        
        # Cache for user preferences
        self.user_preferences = {}
        self.keyword_weights = defaultdict(dict)
        self.emotion_weights = defaultdict(dict)
    
    def _initialize_db(self) -> None:
        """Initialize the SQLite database for storing preferences."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS keyword_preferences (
            user_id TEXT,
            keyword TEXT,
            weight REAL,
            updated_at TEXT,
            PRIMARY KEY (user_id, keyword)
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS emotion_preferences (
            user_id TEXT,
            emotion_type TEXT,
            weight REAL,
            updated_at TEXT,
            PRIMARY KEY (user_id, emotion_type)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def update_preferences(self, user_id: str, clip_metadata: Dict[str, Any], 
                         user_action: str) -> None:
        """
        Update user preferences based on clip feedback.
        
        Args:
            user_id: User ID
            clip_metadata: Metadata about the clip
            user_action: Action taken by user (e.g., 'keep', 'discard', 'share')
        """
        # Extract relevant features from metadata
        keywords = clip_metadata.get('keywords', [])
        emotion_type = clip_metadata.get('emotion_type')
        
        # Action weights
        action_weights = {
            'keep': 0.1,    # Small positive reinforcement
            'favorite': 0.2, # Stronger positive reinforcement
            'share': 0.3,   # Even stronger positive reinforcement
            'discard': -0.1, # Negative reinforcement
            'skip': -0.05   # Slight negative reinforcement
        }
        
        weight_adjustment = action_weights.get(user_action, 0)
        if weight_adjustment == 0:
            return  # No adjustment needed
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        timestamp = datetime.now().isoformat()
        
        # Update keyword preferences
        for keyword in keywords:
            # Get current weight
            cursor.execute(
                "SELECT weight FROM keyword_preferences WHERE user_id = ? AND keyword = ?",
                (user_id, keyword)
            )
            result = cursor.fetchone()
            
            if result:
                current_weight = result[0]
                new_weight = max(0, min(1, current_weight + weight_adjustment))
            else:
                # Initialize with a moderate weight biased by the action
                new_weight = 0.5 + weight_adjustment
            
            # Update database
            cursor.execute(
                "INSERT OR REPLACE INTO keyword_preferences VALUES (?, ?, ?, ?)",
                (user_id, keyword, new_weight, timestamp)
            )
            
            # Update cache
            self.keyword_weights[user_id][keyword] = new_weight
        
        # Update emotion preferences
        if emotion_type:
            # Get current weight
            cursor.execute(
                "SELECT weight FROM emotion_preferences WHERE user_id = ? AND emotion_type = ?",
                (user_id, emotion_type)
            )
            result = cursor.fetchone()
            
            if result:
                current_weight = result[0]
                new_weight = max(0, min(1, current_weight + weight_adjustment))
            else:
                # Initialize with a moderate weight biased by the action
                new_weight = 0.5 + weight_adjustment
            
            # Update database
            cursor.execute(
                "INSERT OR REPLACE INTO emotion_preferences VALUES (?, ?, ?, ?)",
                (user_id, emotion_type, new_weight, timestamp)
            )
            
            # Update cache
            self.emotion_weights[user_id][emotion_type] = new_weight
        
        conn.commit()
        conn.close()
    
    def get_keyword_preference(self, user_id: str, keyword: str) -> float:
        """
        Get the preference weight for a specific keyword.
        
        Args:
            user_id: User ID
            keyword: The keyword to check
            
        Returns:
            Preference weight (0.0-1.0)
        """
        # Check cache first
        if user_id in self.keyword_weights and keyword in self.keyword_weights[user_id]:
            return self.keyword_weights[user_id][keyword]
        
        # If not in cache, check database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT weight FROM keyword_preferences WHERE user_id = ? AND keyword = ?",
            (user_id, keyword)
        )
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            # Update cache and return
            self.keyword_weights[user_id][keyword] = result[0]
            return result[0]
        
        # Default to neutral preference
        return 0.5
    
    def get_emotion_preference(self, user_id: str, emotion_type: str) -> float:
        """
        Get the preference weight for a specific emotion type.
        
        Args:
            user_id: User ID
            emotion_type: Type of emotion (e.g., 'positive', 'negative')
            
        Returns:
            Preference weight (0.0-1.0)
        """
        # Check cache first
        if user_id in self.emotion_weights and emotion_type in self.emotion_weights[user_id]:
            return self.emotion_weights[user_id][emotion_type]
        
        # If not in cache, check database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT weight FROM emotion_preferences WHERE user_id = ? AND emotion_type = ?",
            (user_id, emotion_type)
        )
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            # Update cache and return
            self.emotion_weights[user_id][emotion_type] = result[0]
            return result[0]
        
        # Default to neutral preference
        return 0.5
    
    def rank_suggestions(self, user_id: str, suggestions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank clip suggestions based on user preferences.
        
        Args:
            user_id: User ID
            suggestions: List of clip suggestions
            
        Returns:
            Ranked list of suggestions
        """
        if not suggestions:
            return suggestions
            
        scored_suggestions = []
        
        for suggestion in suggestions:
            score = self._calculate_suggestion_score(user_id, suggestion)
            scored_suggestions.append((score, suggestion))
        
        # Sort by score (descending)
        scored_suggestions.sort(reverse=True)
        
        # Return just the suggestions
        return [item[1] for item in scored_suggestions]
    
    def _calculate_suggestion_score(self, user_id: str, suggestion: Dict[str, Any]) -> float:
        """Calculate a score for a suggestion based on user preferences."""
        score = 0.5  # Base score
        
        # Factor in keywords
        keywords = suggestion.get('keywords', [])
        if keywords:
            keyword_score = sum(self.get_keyword_preference(user_id, kw) for kw in keywords) / len(keywords)
            score += keyword_score * 0.3  # 30% weight for keywords
        
        # Factor in emotion type
        emotion_type = suggestion.get('emotion_type')
        if emotion_type:
            emotion_score = self.get_emotion_preference(user_id, emotion_type)
            score += emotion_score * 0.3  # 30% weight for emotion
        
        # Factor in suggestion confidence
        confidence = suggestion.get('confidence', 0.5)
        score += confidence * 0.2  # 20% weight for confidence
        
        # Normalize to 0-1 range
        return min(1.0, max(0.0, score))


class SuggestionOptimizer:
    """
    Optimizes clip suggestions based on feedback and metrics.
    
    Features:
    - Track success metrics for clips
    - Use reinforcement learning to improve suggestion algorithm
    - Support A/B testing of different suggestion strategies
    """
    
    def __init__(self, db_path: str = "optimization.db"):
        """
        Initialize the suggestion optimizer.
        
        Args:
            db_path: Path to the SQLite database for storing optimization data
        """
        self.db_path = db_path
        
        # Optimization parameters
        self.strategy_weights = {
            'keywords': 0.3,
            'emotion': 0.3,
            'engagement': 0.2,
            'duration': 0.1,
            'audio_quality': 0.1
        }
        
        self.learning_rate = 0.05
        self.exploration_rate = 0.1  # For A/B testing
        
        self._initialize_db()
    
    def _initialize_db(self) -> None:
        """Initialize the SQLite database for storing optimization data."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS clip_metrics (
            clip_id TEXT PRIMARY KEY,
            views INTEGER DEFAULT 0,
            shares INTEGER DEFAULT 0,
            rating REAL DEFAULT 0,
            created_at TEXT,
            updated_at TEXT
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS strategy_performance (
            strategy TEXT PRIMARY KEY,
            weight REAL,
            success_rate REAL,
            sample_count INTEGER,
            updated_at TEXT
        )
        ''')
        
        # Initialize strategy performance if needed
        for strategy, weight in self.strategy_weights.items():
            cursor.execute(
                "INSERT OR IGNORE INTO strategy_performance VALUES (?, ?, ?, ?, ?)",
                (strategy, weight, 0.5, 0, datetime.now().isoformat())
            )
        
        conn.commit()
        conn.close()
    
    def record_metrics(self, clip_id: str, metrics: Dict[str, Any]) -> None:
        """
        Record performance metrics for a clip.
        
        Args:
            clip_id: Clip identifier
            metrics: Dictionary of metrics (views, shares, rating, etc.)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if clip exists
        cursor.execute("SELECT 1 FROM clip_metrics WHERE clip_id = ?", (clip_id,))
        exists = cursor.fetchone() is not None
        
        timestamp = datetime.now().isoformat()
        
        if exists:
            # Update existing metrics
            update_parts = []
            update_values = []
            
            for key, value in metrics.items():
                if key in ('views', 'shares', 'rating'):
                    update_parts.append(f"{key} = {key} + ?")
                    update_values.append(value)
            
            if update_parts:
                update_parts.append("updated_at = ?")
                update_values.append(timestamp)
                
                query = f"UPDATE clip_metrics SET {', '.join(update_parts)} WHERE clip_id = ?"
                cursor.execute(query, tuple(update_values + [clip_id]))
        else:
            # Insert new metrics
            views = metrics.get('views', 0)
            shares = metrics.get('shares', 0)
            rating = metrics.get('rating', 0)
            
            cursor.execute(
                "INSERT INTO clip_metrics VALUES (?, ?, ?, ?, ?, ?)",
                (clip_id, views, shares, rating, timestamp, timestamp)
            )
        
        conn.commit()
        conn.close()
    
    def calculate_clip_success(self, clip_id: str) -> float:
        """
        Calculate a success score for a clip based on its metrics.
        
        Args:
            clip_id: Clip identifier
            
        Returns:
            Success score (0.0-1.0)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT views, shares, rating FROM clip_metrics WHERE clip_id = ?",
            (clip_id,)
        )
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return 0.0
            
        views, shares, rating = result
        
        # Normalize each metric
        # Views: 0-100 → 0-1 (caps at 100 views)
        norm_views = min(views / 100.0, 1.0)
        
        # Shares: 0-10 → 0-1 (caps at 10 shares)
        norm_shares = min(shares / 10.0, 1.0)
        
        # Rating is already 0-5, normalize to 0-1
        norm_rating = rating / 5.0
        
        # Weight the metrics
        # Shares are the strongest signal of quality
        # Rating is next
        # Views are weakest (just watching doesn't mean it was good)
        success_score = (
            norm_views * 0.2 +
            norm_shares * 0.5 +
            norm_rating * 0.3
        )
        
        return success_score
    
    def update_strategy_weights(self, clip_id: str, strategies_used: Dict[str, float]) -> None:
        """
        Update strategy weights based on clip success.
        
        Args:
            clip_id: Clip identifier
            strategies_used: Dictionary mapping strategies to their contribution (0-1)
        """
        # Calculate clip success
        success_score = self.calculate_clip_success(clip_id)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        timestamp = datetime.now().isoformat()
        
        # Update each strategy's performance
        for strategy, contribution in strategies_used.items():
            if contribution <= 0:
                continue  # Skip strategies that weren't used
                
            # Get current performance data
            cursor.execute(
                "SELECT weight, success_rate, sample_count FROM strategy_performance WHERE strategy = ?",
                (strategy,)
            )
            
            result = cursor.fetchone()
            if not result:
                continue
                
            current_weight, current_success, sample_count = result
            
            # Update success rate with new data point
            if sample_count > 0:
                new_success_rate = (current_success * sample_count + success_score) / (sample_count + 1)
            else:
                new_success_rate = success_score
                
            new_sample_count = sample_count + 1
            
            # Adjust weight based on success rate
            # More successful strategies get higher weights
            weight_adjustment = self.learning_rate * (new_success_rate - 0.5) * contribution
            new_weight = max(0.1, min(0.9, current_weight + weight_adjustment))
            
            # Update database
            cursor.execute(
                "UPDATE strategy_performance SET weight = ?, success_rate = ?, sample_count = ?, updated_at = ? WHERE strategy = ?",
                (new_weight, new_success_rate, new_sample_count, timestamp, strategy)
            )
            
            # Update local cache
            self.strategy_weights[strategy] = new_weight
        
        # Normalize weights to sum to 1
        total = sum(self.strategy_weights.values())
        if total > 0:
            for strategy in self.strategy_weights:
                self.strategy_weights[strategy] /= total
        
        conn.commit()
        conn.close()
    
    def get_strategy_weights(self) -> Dict[str, float]:
        """
        Get the current strategy weights.
        
        Returns:
            Dictionary mapping strategies to their weights
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT strategy, weight FROM strategy_performance")
        
        results = cursor.fetchall()
        conn.close()
        
        # Update local cache
        self.strategy_weights = {strategy: weight for strategy, weight in results}
        
        # Normalize weights
        total = sum(self.strategy_weights.values())
        if total > 0:
            for strategy in self.strategy_weights:
                self.strategy_weights[strategy] /= total
        
        return dict(self.strategy_weights)
    
    def select_ab_test_variant(self) -> str:
        """
        Select a variant for A/B testing.
        
        Returns:
            Strategy variant to test
        """
        # Simple exploration vs exploitation
        if np.random.random() < self.exploration_rate:
            # Exploration: try a random strategy with higher weight
            strategies = list(self.strategy_weights.keys())
            return np.random.choice(strategies)
        else:
            # Exploitation: use the best performing strategy
            strategies = list(self.strategy_weights.items())
            strategies.sort(key=lambda x: x[1], reverse=True)
            return strategies[0][0]


class ModelFinetuner:
    """
    Fine-tunes audio models based on user content.
    
    Features:
    - Collect audio samples for training
    - Periodically fine-tune models
    - Track model versions and performance
    """
    
    def __init__(self, 
                models_dir: str = "user_models",
                samples_dir: str = "training_samples"):
        """
        Initialize the model fine-tuner.
        
        Args:
            models_dir: Directory to store fine-tuned models
            samples_dir: Directory to store training samples
        """
        self.models_dir = models_dir
        self.samples_dir = samples_dir
        
        # Create directories if they don't exist
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(samples_dir, exist_ok=True)
        
        # Track collected samples
        self.sample_counts = defaultdict(int)
        
        # Define thresholds for fine-tuning
        self.min_samples_required = 50
        self.fine_tuning_interval = 86400  # Once per day
        self.last_fine_tune_time = {}
    
    def collect_sample(self, user_id: str, audio_data: np.ndarray, 
                      sample_rate: int, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Collect an audio sample for training.
        
        Args:
            user_id: User ID
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            metadata: Optional metadata about the audio
            
        Returns:
            Path to the saved sample file
        """
        # Create user directory if it doesn't exist
        user_dir = os.path.join(self.samples_dir, user_id)
        os.makedirs(user_dir, exist_ok=True)
        
        # Generate a unique filename
        timestamp = int(time.time())
        sample_id = f"{timestamp}_{self.sample_counts[user_id]}"
        audio_path = os.path.join(user_dir, f"{sample_id}.npy")
        
        # Save the audio data
        np.save(audio_path, audio_data)
        
        # Save metadata if provided
        if metadata:
            metadata_path = os.path.join(user_dir, f"{sample_id}.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
        
        # Increment sample count
        self.sample_counts[user_id] += 1
        
        # Check if we should fine-tune
        self._check_fine_tuning_trigger(user_id)
        
        return audio_path
    
    def _check_fine_tuning_trigger(self, user_id: str) -> None:
        """Check if fine-tuning should be triggered for a user."""
        # Check sample count
        if self.sample_counts[user_id] < self.min_samples_required:
            return
            
        # Check time interval
        current_time = time.time()
        last_time = self.last_fine_tune_time.get(user_id, 0)
        
        if current_time - last_time < self.fine_tuning_interval:
            return
            
        # Trigger fine-tuning in a background thread
        logger.info(f"Triggering fine-tuning for user {user_id}")
        
        # In a real implementation, this would be done asynchronously
        # For now, we'll just log it
        logger.info(f"Fine-tuning would be scheduled for user {user_id} with {self.sample_counts[user_id]} samples")
    
    def fine_tune_audio_model(self, user_id: str, model_type: str = "transcription") -> Optional[str]:
        """
        Fine-tune an audio model for a specific user.
        
        Args:
            user_id: User ID
            model_type: Type of model to fine-tune ('transcription', 'emotion', etc.)
            
        Returns:
            Path to the fine-tuned model or None if fine-tuning failed
        """
        # This is a placeholder that would be implemented in a real system
        # Fine-tuning large models requires significant compute resources
        
        user_dir = os.path.join(self.samples_dir, user_id)
        
        # Check if we have enough samples
        sample_files = [f for f in os.listdir(user_dir) if f.endswith('.npy')]
        if len(sample_files) < self.min_samples_required:
            logger.warning(f"Not enough samples for user {user_id} to fine-tune model")
            return None
        
        # In a real implementation, this would:
        # 1. Load the base model
        # 2. Prepare the training data
        # 3. Run fine-tuning with appropriate parameters
        # 4. Save and validate the fine-tuned model
        
        # For now, we'll just generate a mock path
        model_path = os.path.join(self.models_dir, user_id, model_type, "model.pt")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Create an empty file to simulate the model
        with open(model_path, 'w') as f:
            f.write("# Fine-tuned model placeholder")
        
        # Update last fine-tune time
        self.last_fine_tune_time[user_id] = time.time()
        
        logger.info(f"Fine-tuned {model_type} model for user {user_id}")
        return model_path
    
    def get_user_model_path(self, user_id: str, model_type: str) -> Optional[str]:
        """
        Get the path to a user's fine-tuned model.
        
        Args:
            user_id: User ID
            model_type: Type of model ('transcription', 'emotion', etc.)
            
        Returns:
            Path to the model or None if no model exists
        """
        model_path = os.path.join(self.models_dir, user_id, model_type, "model.pt")
        
        if os.path.exists(model_path):
            return model_path
            
        return None
    
    def cleanup_old_samples(self, user_id: str, max_age_days: int = 30) -> int:
        """
        Clean up old training samples to save space.
        
        Args:
            user_id: User ID
            max_age_days: Maximum age of samples to keep
            
        Returns:
            Number of samples removed
        """
        user_dir = os.path.join(self.samples_dir, user_id)
        if not os.path.exists(user_dir):
            return 0
            
        count = 0
        current_time = time.time()
        max_age_seconds = max_age_days * 86400
        
        for filename in os.listdir(user_dir):
            file_path = os.path.join(user_dir, filename)
            file_age = current_time - os.path.getmtime(file_path)
            
            if file_age > max_age_seconds:
                try:
                    os.remove(file_path)
                    count += 1
                except Exception as e:
                    logger.error(f"Error removing old sample {file_path}: {str(e)}")
        
        return count


# Utility functions

def create_learning_services(
    db_dir: str = "db",
    models_dir: str = "models",
    samples_dir: str = "samples"
) -> Tuple[ClipFeedbackTracker, UserPreferenceModel, SuggestionOptimizer, ModelFinetuner]:
    """
    Create a complete set of learning services.
    
    Args:
        db_dir: Directory for databases
        models_dir: Directory for models
        samples_dir: Directory for training samples
        
    Returns:
        Tuple of (ClipFeedbackTracker, UserPreferenceModel, SuggestionOptimizer, ModelFinetuner)
    """
    os.makedirs(db_dir, exist_ok=True)
    
    feedback_tracker = ClipFeedbackTracker(os.path.join(db_dir, "feedback.db"))
    preference_model = UserPreferenceModel(os.path.join(db_dir, "preferences.db"))
    suggestion_optimizer = SuggestionOptimizer(os.path.join(db_dir, "optimization.db"))
    model_finetuner = ModelFinetuner(models_dir, samples_dir)
    
    return (feedback_tracker, preference_model, suggestion_optimizer, model_finetuner) 