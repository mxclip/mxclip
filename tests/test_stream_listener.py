import os
import pytest
import numpy as np
import tempfile
import wave
from unittest.mock import MagicMock, patch
from mxclip.shared_stream_listener import SharedStreamListener


@pytest.fixture
def sample_wav_file():
    """Create a temporary WAV file for testing."""
    # Create a temporary wave file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_path = temp_file.name
    
    # Generate 1 second of silence at 16kHz
    sample_rate = 16000
    duration = 1.0  # seconds
    samples = np.zeros(int(sample_rate * duration), dtype=np.int16)
    
    # Write to WAV file
    with wave.open(temp_path, 'w') as wav:
        wav.setnchannels(1)  # Mono
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)
        wav.writeframes(samples.tobytes())
    
    yield temp_path
    
    # Clean up
    try:
        os.unlink(temp_path)
    except:
        pass


@patch("mxclip.shared_stream_listener.ffmpeg")
def test_shared_stream_listener_init(mock_ffmpeg):
    """Test that SharedStreamListener initializes correctly."""
    # Setup
    filepath = "test_video.mp4"
    push_audio = MagicMock()
    sample_rate = 16000
    chunk_sec = 0.5
    
    # Execute
    listener = SharedStreamListener(filepath, push_audio, sample_rate, chunk_sec)
    
    # Verify
    assert listener.file == filepath
    assert listener.push_audio == push_audio
    assert listener.sample_rate == sample_rate
    assert listener.chunk_sec == chunk_sec


@patch("mxclip.shared_stream_listener.ffmpeg")
def test_shared_stream_listener_start(mock_ffmpeg, sample_wav_file):
    """Test that start method initializes ffmpeg and processes audio chunks."""
    # Setup
    push_audio = MagicMock()
    sample_rate = 16000
    chunk_sec = 0.5
    listener = SharedStreamListener(sample_wav_file, push_audio, sample_rate, chunk_sec)
    
    # Configure mocks
    mock_process = MagicMock()
    mock_ffmpeg.input.return_value.output.return_value.run_async.return_value = mock_process
    
    # Simulate reading chunks from ffmpeg
    # First call returns data, second call returns empty (end of file)
    mock_process.stdout.read.side_effect = [
        np.zeros(int(sample_rate * chunk_sec * 2), dtype=np.int16).tobytes(),
        b''
    ]
    
    # Execute
    listener.start()
    
    # Verify
    mock_ffmpeg.input.assert_called_once_with(sample_wav_file)
    mock_ffmpeg.input.return_value.output.assert_called_once()
    
    # Check output args
    output_args = mock_ffmpeg.input.return_value.output.call_args[0]
    assert output_args[0] == 'pipe:'
    
    # Verify output kwargs
    output_kwargs = mock_ffmpeg.input.return_value.output.call_args[1]
    assert output_kwargs.get('format') == 's16le'
    assert output_kwargs.get('ac') == 1
    assert output_kwargs.get('ar') == str(sample_rate)
    
    # Verify run_async was called
    mock_ffmpeg.input.return_value.output.return_value.run_async.assert_called_once()
    run_kwargs = mock_ffmpeg.input.return_value.output.return_value.run_async.call_args[1]
    assert run_kwargs.get('pipe_stdout') is True
    
    # Verify push_audio was called with correct data type
    push_audio.assert_called_once()
    args = push_audio.call_args[0]
    assert isinstance(args[0], np.ndarray)
    assert args[0].dtype == np.int16


@patch("mxclip.shared_stream_listener.ffmpeg")
def test_shared_stream_listener_error_handling(mock_ffmpeg):
    """Test that SharedStreamListener handles errors properly."""
    # Setup
    filepath = "nonexistent_file.mp4"
    push_audio = MagicMock()
    listener = SharedStreamListener(filepath, push_audio)
    
    # Configure mock to raise an error
    mock_ffmpeg.input.return_value.output.return_value.run_async.side_effect = RuntimeError("Test error")
    
    # Execute and verify
    with pytest.raises(RuntimeError) as excinfo:
        listener.start()
    
    # Check that the error message is correct
    assert "Test error" in str(excinfo.value) 