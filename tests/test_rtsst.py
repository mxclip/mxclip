import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from mxclip.realtime_stt_service import RTSTTService


@pytest.fixture
def sample_audio():
    """Create a sample audio chunk (silent) for testing."""
    # Create 0.5 seconds of silence at 16kHz (8000 samples)
    return np.zeros(8000, dtype=np.int16)


@patch("mxclip.realtime_stt_service.AudioToTextRecorder")
def test_rtstt_service_init(mock_recorder):
    """Test that RTSTTService initializes correctly."""
    # Setup
    callback = MagicMock()
    
    # Execute
    service = RTSTTService(callback, model_size="tiny.en")
    
    # Verify
    mock_recorder.assert_called_once()
    assert service.text_cb == callback
    
    # Verify the recorder was configured correctly
    recorder_args = mock_recorder.call_args[1]
    assert recorder_args["model"] == "tiny.en"
    assert recorder_args["language"] == "en"
    assert recorder_args["use_microphone"] is False
    assert recorder_args["enable_realtime_transcription"] is True
    assert recorder_args["on_realtime_transcription_update"] == callback


@patch("mxclip.realtime_stt_service.AudioToTextRecorder")
def test_rtstt_service_push(mock_recorder, sample_audio):
    """Test that push method forwards audio to the recorder."""
    # Setup
    mock_recorder_instance = mock_recorder.return_value
    service = RTSTTService(MagicMock())
    
    # Execute
    service.push(sample_audio)
    
    # Verify
    mock_recorder_instance.feed_audio.assert_called_once_with(sample_audio)


@patch("mxclip.realtime_stt_service.AudioToTextRecorder")
def test_rtstt_service_shutdown(mock_recorder):
    """Test that shutdown method calls the recorder's shutdown method."""
    # Setup
    mock_recorder_instance = mock_recorder.return_value
    service = RTSTTService(MagicMock())
    
    # Execute
    service.shutdown()
    
    # Verify
    mock_recorder_instance.shutdown.assert_called_once()


@patch("mxclip.realtime_stt_service.AudioToTextRecorder")
def test_rtstt_service_callback(mock_recorder):
    """Test that transcribed text is correctly passed to the callback."""
    # Setup
    callback = MagicMock()
    
    # Configure the mock recorder
    mock_recorder_instance = mock_recorder.return_value
    
    # Create a service with our callback
    service = RTSTTService(callback)
    
    # Simulate the recorder calling the callback
    recorder_args = mock_recorder.call_args[1]
    callback_func = recorder_args["on_realtime_transcription_update"]
    
    # Execute - simulate recorder calling the callback
    test_text = "This is a test transcription"
    callback_func(test_text)
    
    # Verify
    callback.assert_called_once_with(test_text) 