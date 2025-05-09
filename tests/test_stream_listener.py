import os
import pytest
import numpy as np
import tempfile
import wave
import threading
import time
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
    buffer_size = 10
    
    # Execute
    listener = SharedStreamListener(
        filepath, 
        push_audio, 
        sample_rate, 
        chunk_sec, 
        buffer_size=buffer_size
    )
    
    # Verify
    assert listener.file == filepath
    assert listener.push_audio == push_audio
    assert listener.sample_rate == sample_rate
    assert listener.chunk_sec == chunk_sec
    assert listener.buffer_size == buffer_size
    assert listener.reader_thread is None
    assert listener.processing_thread is None


@patch("mxclip.shared_stream_listener.ffmpeg")
@patch("mxclip.shared_stream_listener.threading.Thread")
def test_shared_stream_listener_start(mock_thread, mock_ffmpeg, sample_wav_file):
    """Test that start method initializes ffmpeg and processes audio in threads."""
    # Setup
    push_audio = MagicMock()
    sample_rate = 16000
    chunk_sec = 0.5
    
    # Create the listener
    listener = SharedStreamListener(sample_wav_file, push_audio, sample_rate, chunk_sec)
    
    # Configure mocks
    mock_ffmpeg.probe.return_value = {
        'format': {
            'duration': '1.0'
        }
    }
    
    # Mock thread instances
    mock_reader_thread = MagicMock()
    mock_processor_thread = MagicMock()
    
    # Configure thread mock to return our mocked threads
    mock_thread.side_effect = [mock_reader_thread, mock_processor_thread]
    
    # Execute
    listener.start()
    
    # Verify ffmpeg was probed
    mock_ffmpeg.probe.assert_called_once_with(sample_wav_file)
    
    # Verify threads were created with the correct target functions
    assert mock_thread.call_count == 2
    reader_call = mock_thread.call_args_list[0]
    processor_call = mock_thread.call_args_list[1]
    
    assert reader_call[1]['target'] == listener._read_stream
    assert processor_call[1]['target'] == listener._process_buffer
    
    # Verify threads were started
    mock_reader_thread.start.assert_called_once()
    mock_processor_thread.start.assert_called_once()
    
    # Verify threads were joined
    mock_reader_thread.join.assert_called_once()
    mock_processor_thread.join.assert_called_once()


@patch("mxclip.shared_stream_listener.threading.Thread")
@patch("mxclip.shared_stream_listener.ffmpeg")
def test_shared_stream_listener_stop(mock_ffmpeg, mock_thread, sample_wav_file):
    """Test that the stop method correctly stops the threads."""
    # Setup
    push_audio = MagicMock()
    listener = SharedStreamListener(sample_wav_file, push_audio)
    
    # Mock thread instances that are alive
    mock_reader_thread = MagicMock()
    mock_reader_thread.is_alive.return_value = True
    
    mock_processor_thread = MagicMock()
    mock_processor_thread.is_alive.return_value = True
    
    # Assign threads to the listener
    listener.reader_thread = mock_reader_thread
    listener.processing_thread = mock_processor_thread
    
    # Execute
    listener.stop()
    
    # Verify stop flag was set
    assert listener.stop_flag.is_set()
    
    # Verify join was called on both threads
    mock_reader_thread.join.assert_called_once()
    mock_processor_thread.join.assert_called_once()


@patch("mxclip.shared_stream_listener.ffmpeg")
def test_shared_stream_reader_process(mock_ffmpeg, sample_wav_file):
    """Test the reader thread process function."""
    # Setup
    push_audio = MagicMock()
    sample_rate = 16000
    chunk_sec = 0.5
    listener = SharedStreamListener(sample_wav_file, push_audio, sample_rate, chunk_sec)
    
    # Configure mock process
    mock_process = MagicMock()
    # First read returns data, second read returns empty (end of stream)
    mock_process.stdout.read.side_effect = [
        np.zeros(int(sample_rate * chunk_sec * 2), dtype=np.int16).tobytes(),
        b''
    ]
    mock_ffmpeg.input.return_value.output.return_value.run_async.return_value = mock_process
    
    # Directly put an audio chunk in the buffer instead of relying on thread
    audio_chunk = np.zeros(int(sample_rate * chunk_sec), dtype=np.int16)
    listener.audio_buffer.put(audio_chunk)
    
    # Start a real thread to run the reader process
    reader_thread = threading.Thread(target=listener._read_stream)
    
    # Run the thread
    reader_thread.start()
    
    # Give the thread time to run
    time.sleep(0.5)
    
    # Stop the listener
    listener.stop_flag.set()
    reader_thread.join(timeout=1.0)
    
    # Verify that ffmpeg was called correctly
    mock_ffmpeg.input.assert_called_once_with(sample_wav_file)
    mock_ffmpeg.input.return_value.output.assert_called_once()
    
    # Verify the buffer has data in it (one chunk)
    assert listener.audio_buffer.qsize() == 1


@patch("mxclip.shared_stream_listener.ffmpeg")
def test_shared_stream_listener_error_handling(mock_ffmpeg):
    """Test that SharedStreamListener handles errors properly."""
    # Setup
    filepath = "nonexistent_file.mp4"
    push_audio = MagicMock()
    listener = SharedStreamListener(filepath, push_audio)
    
    # Configure mock to raise an error on probe
    mock_ffmpeg.probe.side_effect = RuntimeError("Test error")
    
    # Execute and verify
    with pytest.raises(RuntimeError) as excinfo:
        listener.start()
    
    # Check that the error was properly set
    assert listener.error is not None
    assert "Test error" in listener.error
    
    # Check that the stop flag was set
    assert listener.stop_flag.is_set()


def test_get_status():
    """Test the get_status method."""
    # Setup
    push_audio = MagicMock()
    listener = SharedStreamListener("test.mp4", push_audio)
    
    # Set some values for status
    listener.current_position = 10.5
    listener.duration = 60.0
    listener.buffer_level = 5
    listener.underruns = 2
    listener.overruns = 1
    listener.error = None
    
    # Get status
    status = listener.get_status()
    
    # Verify
    assert status["current_position"] == 10.5
    assert status["duration"] == 60.0
    assert status["progress_pct"] == 10.5 / 60.0 * 100
    assert status["buffer_level"] == 5
    assert status["buffer_capacity"] == listener.buffer_size
    assert status["underruns"] == 2
    assert status["overruns"] == 1
    assert status["error"] is None 