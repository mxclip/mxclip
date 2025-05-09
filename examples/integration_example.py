"""
Integration example for MX Clipping components.

This example demonstrates how the TriggerDetector, ConfigLoader, ClipsOrganizer,
and Kimi-Audio components work together with the existing MX Clipping system,
including the new learning capabilities.
"""

import os
import time
import logging
import numpy as np
from typing import Dict, Any, List, Optional

# Import existing MX Clipping components
from mxclip.realtime_stt_service import RTSTTService
from mxclip.shared_stream_listener import SharedStreamListener
from mxclip.clip_service import ClipService
from mxclip.chat_service import MockChatService

# Import new components
from mxclip.trigger_detector import TriggerDetector, TriggerEvent
from mxclip.config_loader import ConfigLoader
from mxclip.clips_organizer import ClipsOrganizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize global flags
kimi_audio_available = False
learning_available = False

# Global variables for components
feedback_tracker = None
preference_model = None
suggestion_optimizer = None
model_finetuner = None
trigger_detectors = {}

# Import Kimi-Audio components with proper error handling
try:
    from mxclip.audio_processor import KimiAudioProcessor
    from mxclip.clip_suggestion import ClipSuggester
    kimi_audio_available = True
except ImportError:
    logger.warning("Kimi-Audio modules not available. Some features will be disabled.")
    
# Import learning capabilities
try:
    from mxclip.learning_module import (
        ClipFeedbackTracker, 
        UserPreferenceModel, 
        SuggestionOptimizer,
        ModelFinetuner,
        create_learning_services
    )
    learning_available = True
except ImportError:
    logger.warning("Learning modules not available. Adaptive learning will be disabled.")

def main(video_path):
    """
    Run the integrated example with all MX Clipping components.
    
    Args:
        video_path: Path to the video file to process
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize configuration loader and load user configs
    config_loader = ConfigLoader(config_dir="configs")
    
    user1_config = config_loader.load_config("user1")
    logging.info(f"Loaded config for user1: {user1_config['keywords']}")
    
    user2_config = config_loader.load_config("user2")
    logging.info(f"Loaded config for user2: {user2_config['keywords']}")
    
    # Initialize core services
    clips_organizer = ClipsOrganizer(base_dir="clips")
    clip_service = ClipService(output_dir="clips")
    
    # Initialize learning services
    feedback_tracker = None
    preference_model = None
    suggestion_optimizer = None
    model_finetuner = None
    learning_enabled = False
    kimi_enabled = False
    audio_processor = None
    
    logging.info("Initializing learning services...")
    try:
        feedback_tracker, preference_model, suggestion_optimizer, model_finetuner = create_learning_services()
        learning_enabled = True
        
        # Preload some sample feedback data for the demo
        _preload_feedback_data(feedback_tracker, ["user1", "user2"])
        logging.info("Learning services initialized successfully")
    except Exception as e:
        logging.error(f"Error initializing learning services: {str(e)}")
    
    # Initialize Kimi-Audio for audio analysis (if available)
    logging.info("Initializing Kimi-Audio processor...")
    try:
        audio_processor = KimiAudioProcessor(model_name="kimi-audio-7b-instruct")
        time.sleep(1)  # Allow time for model to load in background
        kimi_enabled = True
    except Exception as e:
        logging.error(f"Failed to initialize Kimi-Audio components: {str(e)}")
        logging.warning("Continuing without Kimi-Audio capabilities")
    
    # Create trigger detectors for users
    user1_detector = TriggerDetector(
        user_id="user1",
        keywords=user1_config["keywords"],
        enable_repeat_check=user1_config["enable_repeat_check"],
        repeat_window_seconds=user1_config["repeat_window_seconds"],
        repeat_threshold=user1_config["repeat_threshold"],
        enable_chat_check=user1_config["enable_chat_check"],
        chat_activity_threshold=user1_config["chat_activity_threshold"]
    )
    logging.info("Created trigger detector for user1")
    
    user2_detector = TriggerDetector(
        user_id="user2",
        keywords=user2_config["keywords"],
        enable_repeat_check=user2_config["enable_repeat_check"],
        repeat_window_seconds=user2_config["repeat_window_seconds"],
        repeat_threshold=user2_config["repeat_threshold"],
        enable_chat_check=user2_config["enable_chat_check"],
        chat_activity_threshold=user2_config["chat_activity_threshold"]
    )
    logging.info("Created trigger detector for user2")
    
    # If learning is enabled, attach feedback tracker to trigger detectors
    if learning_enabled and feedback_tracker:
        user1_detector.set_learning_module(feedback_tracker)
        user2_detector.set_learning_module(feedback_tracker)
    
    # Initialize chat service
    chat_service = MockChatService(message_interval=1.0)
    
    # Initialize real-time speech-to-text service
    def on_transcription(text, timestamp):
        """Handle transcription events."""
        if text:
            logging.info(f"Transcription [{timestamp:.2f}s]: {text}")
            
            # Pass to user trigger detectors
            user1_detector.process_transcription(timestamp, text)
            user2_detector.process_transcription(timestamp, text)
    
    stt_service = RTSTTService(on_transcription)
    
    # Initialize audio/video stream listener
    stream = SharedStreamListener(
        file=video_path,
        push_audio=lambda audio, timestamp: stt_service.push(audio, timestamp),
        sample_rate=16000,
        chunk_sec=0.5
    )
    
    # Set up callbacks for trigger events
    def on_user1_trigger(trigger_event):
        """Handle triggers from user1."""
        _handle_trigger("user1", "demo_streamer", trigger_event, clip_service, clips_organizer, kimi_enabled, audio_processor)
    
    def on_user2_trigger(trigger_event):
        """Handle triggers from user2."""
        _handle_trigger("user2", "demo_streamer", trigger_event, clip_service, clips_organizer, kimi_enabled, audio_processor)
    
    def on_chat_message(timestamp, message, username):
        """Handle incoming chat messages."""
        logging.info(f"Chat [{timestamp:.2f}s]: {username}: {message}")
        
        # Pass chat to the trigger detectors
        user1_detector.process_chat(timestamp, message, username)
        user2_detector.process_chat(timestamp, message, username)
    
    # Register callbacks
    user1_detector.set_callback(on_user1_trigger)
    user2_detector.set_callback(on_user2_trigger)
    chat_service.set_callback(on_chat_message)
    
    # Start services
    try:
        # Start the chat service
        chat_service.start()
        
        # Start the stream listener
        stream.start()
        
        # Give services time to run
        logging.info(f"Processing video: {video_path}")
        logging.info("Press Ctrl+C to stop...")
        
        # Simple animation to show progress
        chars = "|/-\\"
        i = 0
        start_time = time.time()
        
        try:
            while time.time() - start_time < 30:  # Run for 30 seconds or until video ends
                status = stream.get_status()
                position = status["current_position"]
                duration = status["duration"]
                progress = (position / duration * 100) if duration > 0 else 0
                
                # Print progress
                print(f"\r{chars[i]} Processing: {position:.1f}s / {duration:.1f}s [{progress:.1f}%]", end="")
                i = (i + 1) % len(chars)
                
                # Check if video processing is complete
                if position >= duration and duration > 0:
                    print("\nVideo processing complete!")
                    break
                
                time.sleep(0.5)
            
            print("\nExecution complete")
            
        except KeyboardInterrupt:
            print("\nStopped by user")
        
    finally:
        # Stop all services
        stream.stop()
        chat_service.stop()
        stt_service.shutdown()
        
        if learning_enabled and feedback_tracker is not None:
            # Print learning module stats
            logging.info("Learning module statistics:")
            logging.info(f"Feedback database initialized")
            if hasattr(preference_model, "get_update_count"):
                logging.info(f"User preferences updated: {preference_model.get_update_count()} times")
            if hasattr(suggestion_optimizer, "get_optimization_count"):
                logging.info(f"Suggestion optimizations: {suggestion_optimizer.get_optimization_count()}")

def _preload_feedback_data(feedback_tracker, users):
    """Preload some sample feedback data for the demo."""
    # Add some history data for the demo
    for user_id in users:
        # Simulate past feedback for different trigger types
        if user_id == "user1":
            # User 1 likes clips from "amazing" but not "wow"
            feedback_tracker.track_clip_feedback(
                "prev_clip_1", user_id, "keyword:amazing", True, 
                {"text": "That was amazing", "keyword": "amazing"}
            )
            feedback_tracker.track_clip_feedback(
                "prev_clip_2", user_id, "keyword:amazing", True,
                {"text": "This is amazing work", "keyword": "amazing"}
            )
            feedback_tracker.track_clip_feedback(
                "prev_clip_3", user_id, "keyword:wow", False,
                {"text": "Wow that was unexpected", "keyword": "wow"}
            )
            feedback_tracker.track_clip_feedback(
                "prev_clip_4", user_id, "keyword:wow", False,
                {"text": "Wow I didn't see that coming", "keyword": "wow"}
            )
            
            # User 1 likes repetition triggers
            feedback_tracker.track_clip_feedback(
                "prev_clip_5", user_id, "repetition:let's go", True,
                {"text": "let's go", "count": 3}
            )
            
        elif user_id == "user2":
            # User 2 likes clips from "cool" but not "interesting"
            feedback_tracker.track_clip_feedback(
                "prev_clip_6", user_id, "keyword:cool", True,
                {"text": "That was really cool", "keyword": "cool"}
            )
            feedback_tracker.track_clip_feedback(
                "prev_clip_7", user_id, "keyword:cool", True,
                {"text": "So cool how that works", "keyword": "cool"}
            )
            feedback_tracker.track_clip_feedback(
                "prev_clip_8", user_id, "keyword:interesting", False,
                {"text": "That's interesting", "keyword": "interesting"}
            )
            
            # User 2 doesn't like chat activity triggers
            feedback_tracker.track_clip_feedback(
                "prev_clip_9", user_id, "chat_activity", False,
                {"message_count": 16}
            )
            feedback_tracker.track_clip_feedback(
                "prev_clip_10", user_id, "chat_activity", False,
                {"message_count": 12}
            )
    
    # Analyze patterns to update trigger adjustments
    feedback_tracker.analyze_feedback_patterns()
    logger.info("Preloaded sample feedback data for demo")

def process_clip_feedback(clip_id: str, user_id: str, feedback: str, 
                          trigger_reason: str, metadata: Dict[str, Any]) -> None:
    """Process user feedback on a clip."""
    # Access global variables
    global learning_available, trigger_detectors
    global feedback_tracker, preference_model, suggestion_optimizer
    
    if not learning_available:
        return
            
    logger.info(f"Processing feedback for clip {clip_id}: {feedback}")
    
    # Map feedback string to boolean for feedback tracker
    is_kept = feedback in ["keep", "favorite", "share"]
    
    # Track the feedback
    feedback_tracker.track_clip_feedback(
        clip_id=clip_id,
        user_id=user_id,
        reason=trigger_reason,
        is_kept=is_kept,
        metadata=metadata
    )
    
    # Update user preferences
    preference_model.update_preferences(
        user_id=user_id,
        clip_metadata=metadata,
        user_action=feedback
    )
    
    # Record metrics for suggestion optimization
    metrics = {
        "views": 1,
        "shares": 1 if feedback == "share" else 0,
        "rating": 5.0 if feedback == "favorite" else (
                  4.0 if feedback == "keep" else (
                  2.0 if feedback == "skip" else 1.0))
    }
    
    suggestion_optimizer.record_metrics(clip_id=clip_id, metrics=metrics)
    
    # Update trigger detector with new learning
    if user_id in trigger_detectors:
        trigger_detectors[user_id].update_from_feedback()
        
    logger.info(f"Learning updated for user {user_id}")

def _handle_trigger(user_id, streamer_id, trigger_event, clip_service, clips_organizer, kimi_enabled, audio_processor):
    """
    Handle a trigger event by creating a clip.
    
    Args:
        user_id: User ID
        streamer_id: Streamer ID
        trigger_event: The trigger event
        clip_service: ClipService instance
        clips_organizer: ClipsOrganizer instance
        kimi_enabled: Whether Kimi-Audio is enabled
        audio_processor: KimiAudioProcessor instance if enabled
    """
    logging.info(f"Trigger for {user_id}: {trigger_event.reason} at {trigger_event.timestamp:.2f}s")
    
    # Generate clip paths
    video_path, metadata_path = clips_organizer.get_clip_path(
        user_id=user_id,
        streamer_id=streamer_id,
        timestamp=trigger_event.timestamp
    )
    
    # Calculate clip boundaries (15 seconds before and after the trigger)
    clip_start = max(0, trigger_event.timestamp - 15.0)
    clip_end = trigger_event.timestamp + 15.0
    
    # Create transcript placeholder
    transcript = []
    
    try:
        # Create the clip
        clip_service.create_clip(
            video_path="test_data/test_video_with_audio.mp4",  # Use the same path for demo
            center_ts=trigger_event.timestamp,
            reason=trigger_event.reason,
            subtitles=None,
            metadata=trigger_event.metadata,
            output_path=video_path,
            metadata_path=metadata_path
        )
        
        # Generate metadata
        metadata = clips_organizer.generate_metadata(
            user_id=user_id,
            streamer_id=streamer_id,
            trigger_time=trigger_event.timestamp,
            clip_start=clip_start,
            clip_end=clip_end,
            trigger_reason=trigger_event.reason,
            transcript=transcript
        )
        
        # Save metadata
        clips_organizer.save_clip_metadata(metadata_path, metadata)
        
        logging.info(f"Created clip for {user_id} at {video_path}")
        
    except Exception as e:
        logging.error(f"Error creating clip: {str(e)}")

if __name__ == "__main__":
    import sys
    import torch  # Import for cuda availability check
    
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <video_file>")
        sys.exit(1)
    
    main(sys.argv[1]) 