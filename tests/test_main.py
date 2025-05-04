"""
Tests for the main CLI module.
"""

import sys
import argparse
import unittest.mock as mock
import pytest
from io import StringIO

from mxclip.main import (
    cli, _add_subcommands, _run_record, _run_video, _run_analyze, _on_clip_created
)


@pytest.fixture
def mock_parser():
    """Create a mock ArgumentParser for testing."""
    parser = argparse.ArgumentParser()
    _add_subcommands(parser)
    return parser


class TestMainCLI:
    """Test suite for main CLI module."""
    
    @mock.patch('mxclip.main.RTSTTService')
    @mock.patch('mxclip.main.SharedStreamListener')
    def test_run_video(self, mock_listener, mock_stt, capfd):
        """Test _run_video function."""
        # Set up mocks
        mock_stt_instance = mock_stt.return_value
        mock_listener_instance = mock_listener.return_value
        
        # Call the function
        _run_video("test_video.mp4")
        
        # Capture stdout
        out, err = capfd.readouterr()
        
        # Check service was initialized correctly
        mock_stt.assert_called_once()
        mock_listener.assert_called_once_with("test_video.mp4", push_audio=mock_stt_instance.push)
        
        # Check listener was started
        mock_listener_instance.start.assert_called_once()
        
        # Check cleanup happened
        mock_stt_instance.shutdown.assert_called_once()
        
        # Check output
        assert "Transcribing audio from test_video.mp4" in out
    
    @mock.patch('mxclip.main.STTService')
    def test_run_record(self, mock_stt, capfd):
        """Test _run_record function."""
        # Set up mocks
        mock_stt_instance = mock_stt.return_value
        
        # Call the function
        _run_record(10)
        
        # Capture stdout
        out, err = capfd.readouterr()
        
        # Check service was initialized correctly
        mock_stt.assert_called_once()
        
        # Check recording was called with correct duration
        mock_stt_instance.demo_record.assert_called_once_with(10)
        
        # Check output
        assert "Recording for 10 seconds" in out
    
    @mock.patch('mxclip.main.signal')
    @mock.patch('mxclip.main.initialize_metrics')
    @mock.patch('mxclip.main.ClipService')
    @mock.patch('mxclip.main.UserProcessorManager')
    @mock.patch('mxclip.main.MockChatService')
    @mock.patch('mxclip.main.ChatTrigger')
    @mock.patch('mxclip.main.RTSTTService')
    @mock.patch('mxclip.main.SharedStreamListener')
    def test_run_analyze(
        self, mock_listener, mock_stt, mock_chat_trigger, 
        mock_chat_service, mock_user_manager, mock_clip_service,
        mock_metrics, mock_signal
    ):
        """Test _run_analyze function."""
        # Set up args
        args = mock.Mock()
        args.video = "test_video.mp4"
        args.output_dir = "test_clips"
        args.chat_freq = 0.5
        args.metrics_port = 8001
        
        # Set up mocks
        mock_metrics_instance = mock_metrics.return_value
        mock_clip_service_instance = mock_clip_service.return_value
        mock_user_manager_instance = mock_user_manager.return_value
        mock_chat_service_instance = mock_chat_service.return_value
        mock_chat_trigger_instance = mock_chat_trigger.return_value
        mock_stt_instance = mock_stt.return_value
        mock_listener_instance = mock_listener.return_value
        
        # Pre-check for signal handler
        mock_signal.SIGINT = signal_type = mock.sentinel.sigint
        
        # Call the function
        _run_analyze(args)
        
        # Check services were initialized correctly
        mock_metrics.assert_called_once_with(port=8001)
        mock_clip_service.assert_called_once_with(output_dir="test_clips")
        mock_user_manager.assert_called_once()
        mock_chat_service.assert_called_once_with(message_interval=0.5)
        mock_chat_trigger.assert_called_once_with(window_size=5.0, threshold=2.0)
        mock_stt.assert_called_once()
        
        # Check signal handler was set
        mock_signal.signal.assert_called_once_with(signal_type, mock.ANY)
        
        # Check callbacks were set
        mock_chat_service_instance.set_callback.assert_called_once()
        mock_chat_trigger_instance.set_callback.assert_called_once()
        
        # Check services were started
        mock_chat_service_instance.start.assert_called_once()
        mock_listener.assert_called_once_with("test_video.mp4", push_audio=mock_stt_instance.push)
        mock_listener_instance.start.assert_called_once()
        
        # Check cleanup
        mock_chat_service_instance.stop.assert_called_once()
        mock_user_manager_instance.stop_all.assert_called_once()
        mock_stt_instance.shutdown.assert_called_once()
    
    @mock.patch('mxclip.main.get_metrics_service')
    def test_on_clip_created(self, mock_get_metrics, capfd):
        """Test _on_clip_created function."""
        # Setup mock metrics
        mock_metrics = mock_get_metrics.return_value
        
        # Setup clip info
        clip_path = "test_clips/test_clip.mp4"
        clip_info = {
            "reason": "chat_spike",
            "user_id": "test_user",
            "processing_time": 2.5
        }
        
        # Call the function
        _on_clip_created(clip_path, clip_info)
        
        # Capture stdout
        out, err = capfd.readouterr()
        
        # Check metrics were recorded
        mock_metrics.record_clip_created.assert_called_once_with(
            reason="chat_spike",
            user_id="test_user",
            processing_time=2.5
        )
        
        # Check clip precision was updated
        mock_metrics.update_clip_precision.assert_called_once_with(relevant=True)
        
        # Check output
        assert "Created: test_clips/test_clip.mp4" in out
        assert "Reason: chat_spike" in out
        assert "User: test_user" in out
    
    @mock.patch('mxclip.main._run_record')
    @mock.patch('mxclip.main._run_video')
    @mock.patch('mxclip.main._run_analyze')
    @mock.patch('mxclip.main.argparse.ArgumentParser.parse_args')
    def test_cli_record(self, mock_parse_args, mock_analyze, mock_video, mock_record):
        """Test CLI record command."""
        # Setup mock args
        args = mock.Mock()
        args.cmd = "record"
        args.duration = 15
        mock_parse_args.return_value = args
        
        # Call the function
        cli()
        
        # Check appropriate function was called
        mock_record.assert_called_once_with(15)
        mock_video.assert_not_called()
        mock_analyze.assert_not_called()
    
    @mock.patch('mxclip.main._run_record')
    @mock.patch('mxclip.main._run_video') 
    @mock.patch('mxclip.main._run_analyze')
    @mock.patch('mxclip.main.argparse.ArgumentParser.parse_args')
    def test_cli_video(self, mock_parse_args, mock_analyze, mock_video, mock_record):
        """Test CLI video command."""
        # Setup mock args
        args = mock.Mock()
        args.cmd = "video"
        args.file = "test_video.mp4"
        mock_parse_args.return_value = args
        
        # Call the function
        cli()
        
        # Check appropriate function was called
        mock_video.assert_called_once_with("test_video.mp4")
        mock_record.assert_not_called()
        mock_analyze.assert_not_called()
    
    @mock.patch('mxclip.main._run_record')
    @mock.patch('mxclip.main._run_video')
    @mock.patch('mxclip.main._run_analyze')
    @mock.patch('mxclip.main.argparse.ArgumentParser.parse_args')
    def test_cli_analyze(self, mock_parse_args, mock_analyze, mock_video, mock_record):
        """Test CLI analyze command."""
        # Setup mock args
        args = mock.Mock()
        args.cmd = "analyze"
        mock_parse_args.return_value = args
        
        # Call the function
        cli()
        
        # Check appropriate function was called
        mock_analyze.assert_called_once_with(args)
        mock_record.assert_not_called()
        mock_video.assert_not_called() 