import os
import json
import pytest
import tempfile
import shutil
from unittest.mock import MagicMock, patch
import numpy as np

from mxclip.clip_service import ClipService


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for clip output."""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    
    # Clean up
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_video_info():
    """Create mock video information for ffmpeg.probe."""
    return {
        'format': {
            'duration': '120.0',  # 2 minutes video
            'bit_rate': '1000000',
            'size': '15000000'
        },
        'streams': [
            {
                'codec_type': 'video',
                'width': 1280,
                'height': 720,
                'avg_frame_rate': '30/1'
            },
            {
                'codec_type': 'audio',
                'sample_rate': '44100',
                'channels': 2
            }
        ]
    }


@patch('mxclip.clip_service.ffmpeg')
def test_clip_service_init(mock_ffmpeg, temp_output_dir):
    """Test that ClipService initializes correctly."""
    # Setup
    output_dir = temp_output_dir
    clip_length = 30.0
    pre_padding = 15.0
    post_padding = 15.0
    watermark_path = "logo.png"
    watermark_position = "bottomright"
    watermark_size = 0.2
    max_duration = 30.0
    
    # Execute
    service = ClipService(
        output_dir=output_dir,
        clip_length=clip_length,
        pre_padding=pre_padding,
        post_padding=post_padding,
        watermark_path=watermark_path,
        watermark_position=watermark_position,
        watermark_size=watermark_size,
        max_duration=max_duration
    )
    
    # Verify
    assert service.output_dir == output_dir
    assert service.clip_length == clip_length
    assert service.pre_padding == pre_padding
    assert service.post_padding == post_padding
    assert service.watermark_path == watermark_path
    assert service.watermark_position == watermark_position
    assert service.watermark_size == watermark_size
    assert service.max_duration == max_duration
    
    # Verify directory was created
    assert os.path.exists(output_dir)


@patch('mxclip.clip_service.ffmpeg')
def test_clip_duration_within_limits(mock_ffmpeg, temp_output_dir, mock_video_info):
    """Test that created clips have duration between 29-31 seconds."""
    # Setup
    video_path = "test_video.mp4"
    center_ts = 60.0  # 1 minute into the video
    reason = "test_reason"
    
    # Configure mocks
    mock_ffmpeg.probe.return_value = mock_video_info
    
    # Create a service
    service = ClipService(
        output_dir=temp_output_dir,
        clip_length=30.0,
        pre_padding=15.0,
        post_padding=15.0,
        max_duration=30.0
    )
    
    # Mock the ffmpeg.output().run() call
    output_mock = MagicMock()
    mock_ffmpeg.input.return_value = MagicMock()
    mock_ffmpeg.output.return_value = output_mock
    
    # Execute
    clip_path = service.create_clip(
        video_path=video_path,
        center_ts=center_ts,
        reason=reason
    )
    
    # Verify ffmpeg was called with correct time parameters
    input_args = mock_ffmpeg.input.call_args
    assert input_args[0][0] == video_path
    
    # Get the actual clip start and end time
    start_time = center_ts - service.pre_padding
    end_time = center_ts + service.post_padding
    duration = end_time - start_time
    
    # Verify that the clip duration is within the expected range (29-31 seconds)
    assert 29.0 <= duration <= 31.0
    
    # Check that metadata file was created
    metadata_path = clip_path.replace('.mp4', '.json')
    assert os.path.exists(metadata_path)
    
    # Load and check metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    assert metadata['center_timestamp'] == center_ts
    assert metadata['clip_start'] == start_time
    assert metadata['clip_end'] == end_time
    assert metadata['reason'] == reason
    

@patch('mxclip.clip_service.ffmpeg')
def test_clip_max_duration_enforced(mock_ffmpeg, temp_output_dir, mock_video_info):
    """Test that clips don't exceed the maximum duration."""
    # Setup
    video_path = "test_video.mp4"
    center_ts = 60.0  # 1 minute into the video
    reason = "test_reason"
    
    # Configure mocks
    mock_ffmpeg.probe.return_value = mock_video_info
    
    # Create a service with large padding but strict max duration
    service = ClipService(
        output_dir=temp_output_dir,
        clip_length=60.0,  # Try to create a 60s clip
        pre_padding=30.0,
        post_padding=30.0,
        max_duration=30.0  # But enforce 30s max
    )
    
    # Mock the ffmpeg.output().run() call
    output_mock = MagicMock()
    mock_ffmpeg.input.return_value = MagicMock()
    mock_ffmpeg.output.return_value = output_mock
    
    # Execute
    clip_path = service.create_clip(
        video_path=video_path,
        center_ts=center_ts,
        reason=reason
    )
    
    # Load and check metadata
    metadata_path = clip_path.replace('.mp4', '.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Calculate actual duration
    start_time = metadata['clip_start']
    end_time = metadata['clip_end']
    actual_duration = end_time - start_time
    
    # Verify that the duration respects the max limit
    assert actual_duration <= service.max_duration
    
    # Verify it's close to the max (within 1 second margin)
    assert service.max_duration - 1 <= actual_duration <= service.max_duration + 1


@patch('mxclip.clip_service.ffmpeg')
def test_subtitles_added_to_clip(mock_ffmpeg, temp_output_dir, mock_video_info):
    """Test that subtitles are properly added to the clip."""
    # Setup
    video_path = "test_video.mp4"
    center_ts = 60.0
    reason = "test_reason"
    subtitles = [
        (55.0, 58.0, "This is the first subtitle"),
        (59.0, 62.0, "This is the second subtitle"),
        (65.0, 68.0, "This is the third subtitle")
    ]
    
    # Configure mocks
    mock_ffmpeg.probe.return_value = mock_video_info
    
    # Create a service
    service = ClipService(output_dir=temp_output_dir)
    
    # Mock the ffmpeg calls
    mock_ffmpeg.input.return_value = MagicMock()
    output_mock = MagicMock()
    mock_ffmpeg.output.return_value = output_mock
    
    # Mock tempfile.NamedTemporaryFile
    with patch('mxclip.clip_service.tempfile.NamedTemporaryFile') as mock_tempfile:
        mock_tempfile.return_value.__enter__.return_value.name = "mock_subtitle.srt"
        
        # Execute
        clip_path = service.create_clip(
            video_path=video_path,
            center_ts=center_ts,
            reason=reason,
            subtitles=subtitles
        )
    
    # Verify subtitle file was created and ffmpeg was called with appropriate parameters
    mock_ffmpeg.output.assert_called()
    output_args = mock_ffmpeg.output.call_args
    
    # Check that vf parameter contains subtitles
    assert any('subtitles=' in str(arg) for arg in output_args[1].values()) 