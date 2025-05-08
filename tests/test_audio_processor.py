"""Unit tests for the KimiAudioProcessor class."""

import os
import unittest
from unittest.mock import MagicMock, patch
import tempfile
import json
import sys

import pytest

# Create mock transformers module before imports
class MockTransformers:
    class AutoProcessor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return MagicMock()
    
    class AutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return MagicMock()

# Properly mock the transformers module at the system level
sys.modules['transformers'] = MockTransformers()

# Now patch the specific methods/classes
@patch('mxclip.audio_processor.torch')
class TestKimiAudioProcessor:
    """Test suite for KimiAudioProcessor class."""
    
    def test_initialization(self, mock_torch):
        """Test KimiAudioProcessor initialization."""
        from mxclip.audio_processor import KimiAudioProcessor
        
        # Setup mocks
        mock_torch.cuda.is_available.return_value = True
        
        # Initialize processor
        processor = KimiAudioProcessor(
            model_name="test-model",
            device="cuda",
            use_flash_attn=True
        )
        
        # Verify initialization
        assert processor.model_name == "test-model"
        assert processor.device == "cuda"
        assert processor.model is not None
        assert processor.processor is not None
    
    def test_transcribe_audio(self, mock_torch):
        """Test audio transcription functionality."""
        from mxclip.audio_processor import KimiAudioProcessor
        
        # Setup mocks
        mock_torch.cuda.is_available.return_value = True
        
        # Create a temporary audio file
        with tempfile.NamedTemporaryFile(suffix=".wav") as temp_audio:
            # Initialize processor
            processor = KimiAudioProcessor()
            
            # Mock the model and processor for this specific instance
            processor.model = MagicMock()
            processor.processor = MagicMock()
            processor.model.generate.return_value = [MagicMock()]
            
            # Mock processor decode to return a sample transcription
            sample_transcription = "[00:00:00 - 00:00:05] This is a test transcription."
            processor.processor.decode.return_value = sample_transcription
            
            # Patch the _parse_transcription method to return known segments
            with patch.object(processor, '_parse_transcription', return_value=[
                {"start": 0, "end": 5, "text": "This is a test transcription."}
            ]):
                # Call transcribe_audio
                result = processor.transcribe_audio(temp_audio.name)
                
                # Verify results
                assert result["text"] == sample_transcription
                assert len(result["segments"]) == 1
                assert result["segments"][0]["start"] == 0
                assert result["segments"][0]["end"] == 5
                assert result["segments"][0]["text"] == "This is a test transcription."
    
    def test_analyze_audio_content(self, mock_torch):
        """Test audio content analysis functionality."""
        from mxclip.audio_processor import KimiAudioProcessor
        
        # Setup mocks
        mock_torch.cuda.is_available.return_value = False  # Test CPU path
        
        # Create a temporary audio file
        with tempfile.NamedTemporaryFile(suffix=".wav") as temp_audio:
            # Initialize processor
            processor = KimiAudioProcessor(device="cpu")
            
            # Mock the model and processor for this specific instance
            processor.model = MagicMock()
            processor.processor = MagicMock()
            processor.model.generate.return_value = [MagicMock()]
            
            # Mock processor decode to return a sample analysis
            sample_analysis = "The audio contains a person speaking with excitement. Emotion: Happy. Tone: Enthusiastic."
            processor.processor.decode.return_value = sample_analysis
            
            # Mock _parse_analysis to return expected results
            with patch.object(processor, '_parse_analysis', return_value={
                "full_analysis": sample_analysis,
                "summary": "The audio contains a person speaking with excitement.",
                "emotion": "Happy",
                "tone": "Enthusiastic"
            }):
                # Call analyze_audio_content
                result = processor.analyze_audio_content(temp_audio.name)
                
                # Verify results
                assert result["full_analysis"] == sample_analysis
                assert "summary" in result
                assert result["emotion"] == "Happy"
                assert result["tone"] == "Enthusiastic"
    
    def test_detect_keywords(self, mock_torch):
        """Test keyword detection functionality."""
        from mxclip.audio_processor import KimiAudioProcessor
        
        # Setup mocks
        mock_torch.cuda.is_available.return_value = True
        
        # Mock transcribe_audio to return predefined segments
        with patch.object(KimiAudioProcessor, 'transcribe_audio') as mock_transcribe:
            mock_transcribe.return_value = {
                "text": "This is a test with keywords like highlight and amazing moments",
                "segments": [
                    {"start": 0, "end": 5, "text": "This is a test with keywords"},
                    {"start": 5, "end": 10, "text": "like highlight and amazing moments"}
                ]
            }
            
            # Create a temporary audio file
            with tempfile.NamedTemporaryFile(suffix=".wav") as temp_audio:
                # Initialize processor
                processor = KimiAudioProcessor()
                
                # Call detect_keywords
                result = processor.detect_keywords(
                    temp_audio.name, 
                    keywords=["highlight", "amazing"]
                )
                
                # Verify results
                assert "highlight" in result
                assert "amazing" in result
                assert len(result["highlight"]) == 1
                assert result["highlight"][0]["start"] == 5
                assert result["highlight"][0]["end"] == 10
    
    def test_generate_audio_caption(self, mock_torch):
        """Test audio caption generation functionality."""
        from mxclip.audio_processor import KimiAudioProcessor
        
        # Setup mocks
        mock_torch.cuda.is_available.return_value = True
        
        # Create a temporary audio file
        with tempfile.NamedTemporaryFile(suffix=".wav") as temp_audio:
            # Initialize processor
            processor = KimiAudioProcessor()
            
            # Mock the model and processor for this specific instance
            processor.model = MagicMock()
            processor.processor = MagicMock()
            processor.model.generate.return_value = [MagicMock()]
            
            # Mock processor decode to return a sample caption
            sample_caption = "Please provide a short, concise caption for this audio.A person excitedly discussing a gaming victory."
            processor.processor.decode.return_value = sample_caption
            
            # Call generate_audio_caption
            result = processor.generate_audio_caption(temp_audio.name)
            
            # Verify results - should strip the prompt
            assert result == "A person excitedly discussing a gaming victory."
            
    def test_detect_emotional_content(self, mock_torch):
        """Test emotion detection functionality."""
        from mxclip.audio_processor import KimiAudioProcessor
        
        # Initialize processor
        processor = KimiAudioProcessor()
        
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