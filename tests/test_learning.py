"""
Tests for the learning module.
"""

import os
import time
import tempfile
import pytest
import sqlite3
import numpy as np
from unittest.mock import MagicMock, patch
from datetime import datetime

from mxclip.learning_module import (
    ClipFeedbackTracker, 
    UserPreferenceModel, 
    SuggestionOptimizer,
    ModelFinetuner,
    create_learning_services
)

@pytest.fixture
def temp_db_path():
    """Create a temporary directory for test databases."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield os.path.join(temp_dir, "test.db")

@pytest.fixture
def temp_models_dir():
    """Create a temporary directory for test models."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture
def temp_samples_dir():
    """Create a temporary directory for test samples."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

def test_clip_feedback_tracker_init(temp_db_path):
    """Test ClipFeedbackTracker initialization."""
    tracker = ClipFeedbackTracker(db_path=temp_db_path)
    
    # Verify DB tables were created
    conn = sqlite3.connect(temp_db_path)
    cursor = conn.cursor()
    
    # Check clip_feedback table
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='clip_feedback'")
    assert cursor.fetchone() is not None
    
    # Check trigger_adjustments table
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trigger_adjustments'")
    assert cursor.fetchone() is not None
    
    conn.close()

def test_track_clip_feedback(temp_db_path):
    """Test tracking clip feedback."""
    tracker = ClipFeedbackTracker(db_path=temp_db_path)
    
    # Track a clip that was kept
    tracker.track_clip_feedback(
        clip_id="test_clip_1",
        user_id="user1",
        reason="keyword:test",
        is_kept=True,
        metadata={"keyword": "test"}
    )
    
    # Track a clip that was discarded
    tracker.track_clip_feedback(
        clip_id="test_clip_2",
        user_id="user1",
        reason="keyword:test",
        is_kept=False,
        metadata={"keyword": "test"}
    )
    
    # Verify data was stored
    conn = sqlite3.connect(temp_db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT clip_id, is_kept FROM clip_feedback")
    results = cursor.fetchall()
    conn.close()
    
    assert len(results) == 2
    assert ("test_clip_1", 1) in results
    assert ("test_clip_2", 0) in results

def test_analyze_feedback_patterns(temp_db_path):
    """Test analyzing feedback patterns."""
    tracker = ClipFeedbackTracker(db_path=temp_db_path)
    
    # Add multiple clips for a pattern
    for i in range(5):
        tracker.track_clip_feedback(
            clip_id=f"clip_{i}",
            user_id="user1",
            reason="keyword:success",
            is_kept=True,
            metadata={"keyword": "success"}
        )
    
    for i in range(5, 10):
        tracker.track_clip_feedback(
            clip_id=f"clip_{i}",
            user_id="user1",
            reason="keyword:failure",
            is_kept=False,
            metadata={"keyword": "failure"}
        )
    
    # Analyze patterns
    adjustments = tracker.analyze_feedback_patterns()
    
    # Verify adjustments were made and stored
    assert "user1" in adjustments
    assert "keyword" in adjustments["user1"]
    
    # Success keyword should have lower threshold (more sensitive)
    # Failure keyword should have higher threshold (less sensitive)
    keyword_factor = tracker.get_trigger_adjustment("user1", "keyword")
    assert keyword_factor < 1.0  # Should be more sensitive overall
    
    # Check that values were stored in database
    conn = sqlite3.connect(temp_db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT user_id, trigger_type, adjustment_factor FROM trigger_adjustments")
    results = cursor.fetchall()
    conn.close()
    
    assert len(results) > 0
    assert any(row[0] == "user1" and row[1] == "keyword" for row in results)

def test_user_preference_model(temp_db_path):
    """Test UserPreferenceModel functionality."""
    model = UserPreferenceModel(db_path=temp_db_path)
    
    # Test updating preferences
    model.update_preferences(
        user_id="user1",
        clip_metadata={"keywords": ["exciting", "amazing"], "emotion_type": "positive"},
        user_action="favorite"
    )
    
    # Test retrieving preferences
    exciting_weight = model.get_keyword_preference("user1", "exciting")
    amazing_weight = model.get_keyword_preference("user1", "amazing")
    boring_weight = model.get_keyword_preference("user1", "boring")  # Not in database
    emotion_weight = model.get_emotion_preference("user1", "positive")
    
    # Weights should be increased for favorited items
    assert exciting_weight > 0.5
    assert amazing_weight > 0.5
    assert boring_weight == 0.5  # Default neutral weight
    assert emotion_weight > 0.5
    
    # Test negative feedback
    model.update_preferences(
        user_id="user1",
        clip_metadata={"keywords": ["boring"], "emotion_type": "negative"},
        user_action="discard"
    )
    
    boring_weight = model.get_keyword_preference("user1", "boring")
    negative_weight = model.get_emotion_preference("user1", "negative")
    
    # Weights should be decreased for discarded items
    assert boring_weight < 0.5
    assert negative_weight < 0.5

def test_suggestion_ranking(temp_db_path):
    """Test ranking of suggestions based on preferences."""
    model = UserPreferenceModel(db_path=temp_db_path)
    
    # Set up some preferences
    model.update_preferences(
        user_id="user1",
        clip_metadata={"keywords": ["exciting"], "emotion_type": "positive"},
        user_action="favorite"
    )
    
    model.update_preferences(
        user_id="user1",
        clip_metadata={"keywords": ["boring"], "emotion_type": "negative"},
        user_action="discard"
    )
    
    # Create test suggestions
    suggestions = [
        {
            "id": "1",
            "keywords": ["exciting", "amazing"],
            "emotion_type": "positive",
            "confidence": 0.8
        },
        {
            "id": "2",
            "keywords": ["boring", "standard"],
            "emotion_type": "negative",
            "confidence": 0.6
        },
        {
            "id": "3",
            "keywords": ["standard"],
            "emotion_type": "neutral",
            "confidence": 0.7
        }
    ]
    
    # Rank the suggestions
    ranked = model.rank_suggestions("user1", suggestions)
    
    # Verify ranking - exciting/positive should be first, boring/negative should be last
    assert ranked[0]["id"] == "1"
    assert ranked[-1]["id"] == "2"

def test_suggestion_optimizer(temp_db_path):
    """Test SuggestionOptimizer functionality."""
    optimizer = SuggestionOptimizer(db_path=temp_db_path)
    
    # Record metrics for a clip
    optimizer.record_metrics(
        clip_id="test_clip",
        metrics={"views": 50, "shares": 5, "rating": 4.5}
    )
    
    # Calculate success score
    score = optimizer.calculate_clip_success("test_clip")
    
    # Verify score calculation
    assert 0 <= score <= 1
    assert score > 0.5  # Should be high given good metrics
    
    # Update strategy weights
    optimizer.update_strategy_weights(
        clip_id="test_clip",
        strategies_used={"keywords": 0.7, "emotion": 0.3}
    )
    
    # Get updated weights
    weights = optimizer.get_strategy_weights()
    
    # Verify weights were updated and normalized
    assert "keywords" in weights
    assert "emotion" in weights
    assert abs(sum(weights.values()) - 1.0) < 0.001  # Sum should be close to 1.0

def test_model_finetuner(temp_models_dir, temp_samples_dir):
    """Test ModelFinetuner functionality."""
    finetuner = ModelFinetuner(
        models_dir=temp_models_dir,
        samples_dir=temp_samples_dir
    )
    
    # Create sample audio data
    audio_data = np.random.randn(16000)  # 1 second of random audio
    
    # Collect a sample
    sample_path = finetuner.collect_sample(
        user_id="user1",
        audio_data=audio_data,
        sample_rate=16000,
        metadata={"source": "test"}
    )
    
    # Verify sample was saved
    assert os.path.exists(sample_path)
    
    # Force fine-tuning with a small sample size for testing
    finetuner.min_samples_required = 1
    
    # Fine-tune model
    model_path = finetuner.fine_tune_audio_model(
        user_id="user1",
        model_type="transcription"
    )
    
    # Verify model file was created
    assert os.path.exists(model_path)
    
    # Test get_user_model_path
    retrieved_path = finetuner.get_user_model_path("user1", "transcription")
    assert retrieved_path == model_path
    
    # Test nonexistent model
    assert finetuner.get_user_model_path("nonexistent", "transcription") is None

def test_create_learning_services():
    """Test creating all learning services together."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_dir = os.path.join(temp_dir, "db")
        models_dir = os.path.join(temp_dir, "models")
        samples_dir = os.path.join(temp_dir, "samples")
        
        feedback_tracker, preference_model, suggestion_optimizer, model_finetuner = create_learning_services(
            db_dir=db_dir,
            models_dir=models_dir,
            samples_dir=samples_dir
        )
        
        # Verify all components were created
        assert isinstance(feedback_tracker, ClipFeedbackTracker)
        assert isinstance(preference_model, UserPreferenceModel)
        assert isinstance(suggestion_optimizer, SuggestionOptimizer)
        assert isinstance(model_finetuner, ModelFinetuner)
        
        # Verify directories were created
        assert os.path.exists(db_dir)
        assert os.path.exists(models_dir)
        assert os.path.exists(samples_dir) 