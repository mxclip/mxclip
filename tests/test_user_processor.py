"""
Tests for the UserProcessor module.
"""

import time
import pytest
from unittest.mock import Mock, MagicMock
from mxclip.user_processor import UserProcessor, UserProcessorManager, TriggerEvent

class TestUserProcessor:
    """Test suite for UserProcessor."""
    
    def test_init(self):
        """Test initialization of UserProcessor."""
        # Create a mock clip service
        mock_clip_service = Mock()
        
        # Initialize user processor
        processor = UserProcessor(
            user_id="test_user",
            clip_service=mock_clip_service,
            video_path="test_video.mp4",
            min_interval=5.0,
            max_queue_size=10
        )
        
        # Check initial state
        assert processor.user_id == "test_user"
        assert processor.video_path == "test_video.mp4"
        assert processor.min_interval == 5.0
        assert processor.running is False
        assert processor.triggers_received == 0
        assert processor.clips_created == 0
        assert processor.triggers_dropped == 0
    
    def test_add_trigger(self):
        """Test adding a trigger event."""
        # Create a mock clip service
        mock_clip_service = Mock()
        
        # Initialize user processor
        processor = UserProcessor(
            user_id="test_user",
            clip_service=mock_clip_service,
            video_path="test_video.mp4"
        )
        
        # Create a trigger event
        trigger = TriggerEvent(timestamp=10.0, reason="test_trigger")
        
        # Add the trigger
        result = processor.add_trigger(trigger)
        
        # Check result and stats
        assert result is True
        assert processor.triggers_received == 1
        assert processor.get_stats()["triggers_received"] == 1
    
    def test_get_stats(self):
        """Test getting processor statistics."""
        # Create a mock clip service
        mock_clip_service = Mock()
        
        # Initialize user processor
        processor = UserProcessor(
            user_id="stats_test_user",
            clip_service=mock_clip_service,
            video_path="test_video.mp4"
        )
        
        # Add a few triggers
        for i in range(3):
            trigger = TriggerEvent(timestamp=float(i), reason=f"test_trigger_{i}")
            processor.add_trigger(trigger)
        
        # Get stats
        stats = processor.get_stats()
        
        # Check stats
        assert stats["user_id"] == "stats_test_user"
        assert stats["triggers_received"] == 3
        assert stats["clips_created"] == 0  # No clips created yet
        assert stats["triggers_dropped"] == 0
        assert stats["running"] is False
    
    def test_start_stop(self):
        """Test starting and stopping the processor."""
        # Create a mock clip service
        mock_clip_service = Mock()
        
        # Initialize user processor
        processor = UserProcessor(
            user_id="start_stop_test_user",
            clip_service=mock_clip_service,
            video_path="test_video.mp4"
        )
        
        # Start the processor
        processor.start()
        
        # Check that it's running
        assert processor.running is True
        assert processor.thread is not None
        
        # Stop the processor
        processor.stop()
        
        # Check that it's stopped
        assert processor.running is False
    
    def test_process_trigger(self):
        """Test processing a trigger event."""
        # Create a mock clip service
        mock_clip_service = Mock()
        mock_clip_service.create_clip.return_value = "test_clip.mp4"
        
        # Create a mock callback
        mock_callback = Mock()
        
        # Initialize user processor
        processor = UserProcessor(
            user_id="process_test_user",
            clip_service=mock_clip_service,
            video_path="test_video.mp4",
            callback=mock_callback
        )
        
        # Create a trigger with transcript
        transcript = [{"start": 8.0, "end": 12.0, "text": "Test transcript"}]
        trigger = TriggerEvent(
            timestamp=10.0, 
            reason="test_trigger",
            metadata={"transcript": transcript}
        )
        
        # Start the processor
        processor.start()
        
        # Add the trigger
        processor.add_trigger(trigger)
        
        # Wait for processing
        time.sleep(0.5)
        
        # Stop the processor
        processor.stop()
        
        # Check that clip service was called
        mock_clip_service.create_clip.assert_called_once()
        args, kwargs = mock_clip_service.create_clip.call_args
        
        # Check arguments
        assert kwargs["video_path"] == "test_video.mp4"
        assert kwargs["center_ts"] == 10.0
        assert kwargs["reason"] == "test_trigger"
        assert kwargs["metadata"]["user_id"] == "process_test_user"
        
        # Check subtitles
        subtitles = kwargs["subtitles"]
        assert subtitles[0][0] == 8.0  # start
        assert subtitles[0][1] == 12.0  # end
        assert subtitles[0][2] == "Test transcript"  # text
        
        # Check callback was called
        mock_callback.assert_called_once()
        

class TestUserProcessorManager:
    """Test suite for UserProcessorManager."""
    
    def test_get_processor(self):
        """Test getting or creating a processor."""
        # Create a mock clip service
        mock_clip_service = Mock()
        
        # Initialize manager
        manager = UserProcessorManager(
            clip_service=mock_clip_service,
            video_path="test_video.mp4"
        )
        
        # Get a processor
        processor = manager.get_processor("test_user")
        
        # Check that it's the right type
        assert isinstance(processor, UserProcessor)
        assert processor.user_id == "test_user"
        
        # Get the same processor again
        processor2 = manager.get_processor("test_user")
        
        # Check that it's the same instance
        assert processor is processor2
        
        # Get a different processor
        processor3 = manager.get_processor("another_user")
        
        # Check that it's a different instance
        assert processor is not processor3
        assert processor3.user_id == "another_user"
    
    def test_add_trigger(self):
        """Test adding a trigger through the manager."""
        # Create a mock clip service
        mock_clip_service = Mock()
        
        # Initialize manager
        manager = UserProcessorManager(
            clip_service=mock_clip_service,
            video_path="test_video.mp4"
        )
        
        # Create a trigger
        trigger = TriggerEvent(timestamp=10.0, reason="manager_test_trigger")
        
        # Add the trigger for a new user
        result = manager.add_trigger("manager_test_user", trigger)
        
        # Check result
        assert result is True
        
        # Check that the processor was created
        assert "manager_test_user" in manager.processors
        
        # Check that the trigger was added to the processor
        processor = manager.processors["manager_test_user"]
        assert processor.triggers_received == 1
    
    def test_get_all_stats(self):
        """Test getting stats for all processors."""
        # Create a mock clip service
        mock_clip_service = Mock()
        
        # Initialize manager
        manager = UserProcessorManager(
            clip_service=mock_clip_service,
            video_path="test_video.mp4"
        )
        
        # Create a few processors by adding triggers
        users = ["stats_user1", "stats_user2", "stats_user3"]
        for user_id in users:
            trigger = TriggerEvent(timestamp=10.0, reason=f"{user_id}_trigger")
            manager.add_trigger(user_id, trigger)
        
        # Get all stats
        all_stats = manager.get_all_stats()
        
        # Check stats
        assert len(all_stats) == len(users)
        
        # Check each user's stats
        user_ids = {stats["user_id"] for stats in all_stats}
        assert user_ids == set(users)
        
        # Check each user has one trigger received
        for stats in all_stats:
            assert stats["triggers_received"] == 1
    
    def test_stop_all(self):
        """Test stopping all processors."""
        # Create a mock clip service
        mock_clip_service = Mock()
        
        # Initialize manager
        manager = UserProcessorManager(
            clip_service=mock_clip_service,
            video_path="test_video.mp4"
        )
        
        # Create and start a few processors
        users = ["stop_user1", "stop_user2"]
        for user_id in users:
            processor = manager.get_processor(user_id)
            processor.start()
            assert processor.running is True
        
        # Stop all processors
        manager.stop_all()
        
        # Check that all processors are stopped
        for user_id in users:
            processor = manager.processors[user_id]
            assert processor.running is False 