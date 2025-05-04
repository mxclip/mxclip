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
    
    # Now we check that the wrapper callback is set instead of the original
    assert recorder_args["on_realtime_transcription_update"] != callback
    assert callable(recorder_args["on_realtime_transcription_update"])


@patch("mxclip.realtime_stt_service.AudioToTextRecorder")
def test_rtstt_service_push(mock_recorder):
    """Test that audio is correctly passed to the recorder."""
    # Setup
    mock_recorder_instance = mock_recorder.return_value
    service = RTSTTService(MagicMock())
    
    # Execute - push some fake audio data
    fake_audio = MagicMock()
    service.push(fake_audio)
    
    # Verify
    mock_recorder_instance.feed_audio.assert_called_once_with(fake_audio)


@patch("mxclip.realtime_stt_service.AudioToTextRecorder")
def test_rtstt_service_push_with_timestamp(mock_recorder):
    """Test that audio is correctly passed to the recorder with timestamp."""
    # Setup
    mock_recorder_instance = mock_recorder.return_value
    service = RTSTTService(MagicMock())
    
    # Execute - push some fake audio data with timestamp
    fake_audio = MagicMock()
    timestamp = 12.34
    service.push(fake_audio, timestamp)
    
    # Verify
    mock_recorder_instance.feed_audio.assert_called_once_with(fake_audio)
    assert service.current_timestamp == timestamp


@patch("mxclip.realtime_stt_service.AudioToTextRecorder")
def test_rtstt_service_callback(mock_recorder):
    """Test that transcribed text is correctly passed to the callback."""
    # Setup
    callback = MagicMock()
    
    # Configure the mock recorder
    mock_recorder_instance = mock_recorder.return_value
    
    # Create a service with our callback
    service = RTSTTService(callback)
    
    # Set current timestamp
    test_timestamp = 45.67
    service.current_timestamp = test_timestamp

    # Simulate the recorder calling the callback
    recorder_args = mock_recorder.call_args[1]
    callback_func = recorder_args["on_realtime_transcription_update"]
    
    # Execute - simulate recorder calling the callback
    test_text = "This is a test transcription"
    callback_func(test_text)
    
    # Verify callback is called with text and timestamp
    callback.assert_called_once_with(test_text, test_timestamp)


@patch("mxclip.realtime_stt_service.AudioToTextRecorder")
def test_rtstt_service_shutdown(mock_recorder):
    """Test that shutdown is correctly propagated to the recorder."""
    # Setup
    mock_recorder_instance = mock_recorder.return_value
    service = RTSTTService(MagicMock())
    
    # Execute
    service.shutdown()
    
    # Verify
    mock_recorder_instance.shutdown.assert_called_once() 