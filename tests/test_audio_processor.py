"""Unit tests for the KimiAudioProcessor class."""

import os
import unittest
from unittest.mock import MagicMock, patch
import tempfile
import json

import pytest

# Test with mocked Kimi-Audio imports
@patch('mxclip.audio_processor.torch')
@patch('mxclip.audio_processor.AutoProcessor')
@patch('mxclip.audio_processor.AutoModelForCausalLM')
class TestKimiAudioProcessor:
    """Test suite for KimiAudioProcessor class."""
    
    def test_initialization(self, mock_model_class, mock_processor_class, mock_torch):
        """Test KimiAudioProcessor initialization."""
        from mxclip.audio_processor import KimiAudioProcessor
        
        # Setup mocks
        mock_torch.cuda.is_available.return_value = True
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Initialize processor
        processor = KimiAudioProcessor(
            model_name="test-model",
            device="cuda",
            use_flash_attn=True
        )
        
        # Verify initialization
        mock_processor_class.from_pretrained.assert_called_once_with("test-model")
        mock_model_class.from_pretrained.assert_called_once()
        assert processor.model_name == "test-model"
        assert processor.device == "cuda"
        assert processor.model == mock_model
        assert processor.processor == mock_processor
    
    def test_transcribe_audio(self, mock_model_class, mock_processor_class, mock_torch):
        """Test audio transcription functionality."""
        from mxclip.audio_processor import KimiAudioProcessor
        
        # Setup mocks
        mock_torch.cuda.is_available.return_value = True
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Mock model generation
        mock_outputs = MagicMock()
        mock_model.generate.return_value = [mock_outputs]
        
        # Mock processor decode to return a sample transcription
        sample_transcription = "[00:00:00 - 00:00:05] This is a test transcription."
        mock_processor.decode.return_value = sample_transcription
        
        # Create a temporary audio file
        with tempfile.NamedTemporaryFile(suffix=".wav") as temp_audio:
            # Initialize processor
            processor = KimiAudioProcessor()
            
            # Call transcribe_audio
            result = processor.transcribe_audio(temp_audio.name)
            
            # Verify results
            assert result["text"] == sample_transcription
            assert len(result["segments"]) == 1
            assert result["segments"][0]["start"] == 0
            assert result["segments"][0]["end"] == 5
            assert result["segments"][0]["text"] == "This is a test transcription."
    
    def test_analyze_audio_content(self, mock_model_class, mock_processor_class, mock_torch):
        """Test audio content analysis functionality."""
        from mxclip.audio_processor import KimiAudioProcessor
        
        # Setup mocks
        mock_torch.cuda.is_available.return_value = False  # Test CPU path
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Mock model generation
        mock_outputs = MagicMock()
        mock_model.generate.return_value = [mock_outputs]
        
        # Mock processor decode to return a sample analysis
        sample_analysis = "The audio contains a person speaking with excitement. Emotion: Happy. Tone: Enthusiastic."
        mock_processor.decode.return_value = sample_analysis
        
        # Create a temporary audio file
        with tempfile.NamedTemporaryFile(suffix=".wav") as temp_audio:
            # Initialize processor
            processor = KimiAudioProcessor(device="cpu")
            
            # Call analyze_audio_content
            result = processor.analyze_audio_content(temp_audio.name)
            
            # Verify results
            assert result["full_analysis"] == sample_analysis
            assert "summary" in result
            assert result["emotion"] == "Happy"
            assert result["tone"] == "Enthusiastic"
    
    def test_detect_keywords(self, mock_model_class, mock_processor_class, mock_torch):
        """Test keyword detection functionality."""
        from mxclip.audio_processor import KimiAudioProcessor
        
        # Setup mocks
        mock_torch.cuda.is_available.return_value = True
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = mock_model
        
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
    
    def test_generate_audio_caption(self, mock_model_class, mock_processor_class, mock_torch):
        """Test audio caption generation functionality."""
        from mxclip.audio_processor import KimiAudioProcessor
        
        # Setup mocks
        mock_torch.cuda.is_available.return_value = True
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Mock model generation
        mock_outputs = MagicMock()
        mock_model.generate.return_value = [mock_outputs]
        
        # Mock processor decode to return a sample caption
        sample_caption = "Please provide a short, concise caption for this audio.A person excitedly discussing a gaming victory."
        mock_processor.decode.return_value = sample_caption
        
        # Create a temporary audio file
        with tempfile.NamedTemporaryFile(suffix=".wav") as temp_audio:
            # Initialize processor
            processor = KimiAudioProcessor()
            
            # Call generate_audio_caption
            result = processor.generate_audio_caption(temp_audio.name)
            
            # Verify results - should strip the prompt
            assert result == "A person excitedly discussing a gaming victory." 