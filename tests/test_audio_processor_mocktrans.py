"""Tests for emotion detection in KimiAudioProcessor with mock transformers."""

import os
import unittest
from unittest.mock import MagicMock, patch

import pytest

# Patch KimiAudioProcessor._load_model at module level
@patch('mxclip.audio_processor.KimiAudioProcessor._load_model', MagicMock(return_value=None))
class TestEmotionDetection:
    """Test emotion detection capabilities."""
    
    def test_detect_emotional_content(self):
        """Test the detect_emotional_content method."""
        from mxclip.audio_processor import KimiAudioProcessor
        
        # Create processor with the patched _load_model method
        processor = KimiAudioProcessor()
        
        # Set required mocks
        processor.model = MagicMock()
        processor.processor = MagicMock()
        
        # Test positive emotions
        positive_text = "I'm so excited about this amazing result! It's awesome!"
        pos_result = processor.detect_emotional_content(positive_text)
        assert pos_result["has_emotion"] is True
        assert pos_result["emotion_type"] in ["positive", "positive_surprise"]
        assert "excited" in pos_result["emotion_words"]
        assert "amazing" in pos_result["emotion_words"]
        assert "awesome" in pos_result["emotion_words"]
        assert pos_result["intensity"] > 0.3  # Should have substantial intensity
        
        # Test negative emotions
        negative_text = "I'm so angry and upset about this terrible outcome. It's horrible."
        neg_result = processor.detect_emotional_content(negative_text)
        assert neg_result["has_emotion"] is True
        assert neg_result["emotion_type"] == "negative"
        assert "angry" in neg_result["emotion_words"]
        assert "terrible" in neg_result["emotion_words"]
        assert "horrible" in neg_result["emotion_words"]
        assert neg_result["intensity"] > 0.3  # Should have substantial intensity
        
        # Test mixed emotions
        mixed_text = "I'm excited but also worried about the shocking results."
        mixed_result = processor.detect_emotional_content(mixed_text)
        assert mixed_result["has_emotion"] is True
        assert mixed_result["emotion_type"] in ["mixed", "positive_surprise"]
        assert len(mixed_result["emotion_words"]) >= 2
        
        # Test no emotions
        neutral_text = "The meeting is scheduled for tomorrow at 2pm in room 302."
        neutral_result = processor.detect_emotional_content(neutral_text)
        assert neutral_result["has_emotion"] is False
        assert neutral_result["emotion_type"] is None
        assert len(neutral_result["emotion_words"]) == 0
        assert neutral_result["intensity"] == 0.0
    
    def test_analyze_segments_for_emotion(self):
        """Test the analyze_segments_for_emotion method."""
        from mxclip.audio_processor import KimiAudioProcessor
        
        # Create processor with patched _load_model method
        processor = KimiAudioProcessor()
        
        # Set required mocks
        processor.model = MagicMock()
        processor.processor = MagicMock()
        
        # Test segments with mixed emotional content
        segments = [
            {"start": 0, "end": 5, "text": "Let's begin our regular meeting."},
            {"start": 5, "end": 10, "text": "I'm really excited about our amazing results!"},
            {"start": 10, "end": 15, "text": "We need to discuss the next steps."},
            {"start": 15, "end": 20, "text": "But I'm worried about the terrible feedback we got."}
        ]
        
        analyzed = processor.analyze_segments_for_emotion(segments)
        
        # Check all segments have emotion analysis
        assert len(analyzed) == 4
        for segment in analyzed:
            assert "has_emotion" in segment
            assert "emotion_type" in segment
            assert "emotion_words" in segment
            assert "emotion_intensity" in segment
        
        # Check positive emotion in segment 1
        assert analyzed[1]["has_emotion"] is True
        assert analyzed[1]["emotion_type"] in ["positive", "positive_surprise"]
        assert "excited" in analyzed[1]["emotion_words"]
        assert "amazing" in analyzed[1]["emotion_words"]
        
        # Check negative emotion in segment 3
        assert analyzed[3]["has_emotion"] is True
        assert analyzed[3]["emotion_type"] == "negative"
        assert "terrible" in analyzed[3]["emotion_words"]
        
        # Check neutral segments
        assert analyzed[0]["has_emotion"] is False
        assert analyzed[2]["has_emotion"] is False 