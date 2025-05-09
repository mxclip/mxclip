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

def main(video_file: str):
    """
    Run an integrated example of MX Clipping with new components.
    
    Args:
        video_file: Path to video file to process
    """
    # Access global variables
    global learning_available, kimi_audio_available
    global feedback_tracker, preference_model, suggestion_optimizer, model_finetuner, trigger_detectors
    
    # Step 1: Set up the config loader and load user configs
    config_loader = ConfigLoader(config_dir="configs")
    
    # For demo, we'll use a few predefined users
    users = ["user1", "user2"]
    user_configs = {}
    
    for user_id in users:
        # Update with custom keywords for demo
        if user_id == "user1":
            config_loader.update_config(user_id, {
                "keywords": ["awesome", "amazing", "wow"],
                "enable_repeat_check": True,
                "repeat_threshold": 2
            })
        elif user_id == "user2":
            config_loader.update_config(user_id, {
                "keywords": ["interesting", "cool", "nice"],
                "enable_chat_check": True,
                "chat_activity_threshold": 3
            })
        
        # Load the configs
        user_configs[user_id] = config_loader.load_config(user_id)
        logger.info(f"Loaded config for {user_id}: {user_configs[user_id]['keywords']}")
    
    # Step 2: Set up the clips organizer
    clips_organizer = ClipsOrganizer(base_dir="clips")
    
    # For the demo, we'll use a static streamer ID
    streamer_id = "demo_streamer"
    
    # Step 3: Set up the clip service
    clip_service = ClipService(output_dir="clips")
    
    # Step 4: Initialize learning services if available
    feedback_tracker = None
    preference_model = None 
    suggestion_optimizer = None
    model_finetuner = None
    
    if learning_available:
        try:
            logger.info("Initializing learning services...")
            # Create database and model directories
            os.makedirs("db", exist_ok=True)
            os.makedirs("models", exist_ok=True)
            os.makedirs("samples", exist_ok=True)
            
            # Initialize all learning services
            feedback_tracker, preference_model, suggestion_optimizer, model_finetuner = create_learning_services(
                db_dir="db",
                models_dir="models",
                samples_dir="samples"
            )
            
            logger.info("Learning services initialized successfully")
            
            # Pre-load some sample feedback data for the demo
            _preload_feedback_data(feedback_tracker, users)
        except Exception as e:
            logger.error(f"Failed to initialize learning services: {str(e)}")
            logger.warning("Continuing without adaptive learning capabilities")
            learning_available = False
    
    # Step 5: Initialize Kimi-Audio components if available
    audio_processor = None
    clip_suggester = None
    
    if kimi_audio_available:
        try:
            logger.info("Initializing Kimi-Audio processor...")
            import torch
            audio_processor = KimiAudioProcessor(
                model_name="kimi-audio-7b-instruct",
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            logger.info("Initializing clip suggester...")
            clip_suggester = ClipSuggester(
                audio_processor=audio_processor,
                min_clip_duration=5.0,
                max_clip_duration=30.0
            )
            logger.info("Kimi-Audio components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Kimi-Audio components: {str(e)}")
            logger.warning("Continuing without Kimi-Audio capabilities")
            kimi_audio_available = False
    
    # Step 6: Set up trigger detectors for each user
    trigger_detectors = {}
    
    for user_id, config in user_configs.items():
        # Create a trigger detector with config settings and learning capabilities
        detector = TriggerDetector(
            keywords=config["keywords"],
            enable_repeat_check=config["enable_repeat_check"],
            repeat_window_seconds=config["repeat_window_seconds"],
            repeat_threshold=config["repeat_threshold"],
            enable_chat_check=config["enable_chat_check"],
            chat_activity_threshold=config["chat_activity_threshold"],
            feedback_tracker=feedback_tracker,
            user_id=user_id
        )
        
        # Set up the callback for triggers
        detector.set_callback(lambda event, uid=user_id: on_trigger(event, uid, streamer_id))
        
        trigger_detectors[user_id] = detector
        logger.info(f"Created trigger detector for {user_id}")
    
    # Step 7: Set up the chat service
    chat_service = MockChatService(message_interval=1.0)
    
    # Process chat messages for each user's trigger detector
    def on_chat_message(timestamp: float, message: str, username: str):
        logger.info(f"Chat: {username}: {message}")
        for user_id, detector in trigger_detectors.items():
            detector.process_chat(timestamp, message, username)
    
    chat_service.set_callback(on_chat_message)
    
    # Step 8: Set up the STT service for transcription with improved buffering
    transcript_buffer = []
    emotion_buffer = []  # New buffer for emotion detection results
    
    def on_transcription(text: str, actual_timestamp: Optional[float] = None):
        if not text:
            return
        
        # Use actual audio timestamp if available, otherwise current time
        timestamp = actual_timestamp if actual_timestamp is not None else time.time()
        logger.info(f"Transcript: {text}")
        
        # Estimate the duration based on text length (more accurate than fixed multiplier)
        def estimate_duration(text: str) -> float:
            """Estimate speech duration based on word count and characters."""
            words = len(text.split())
            chars = len(text)
            # Average English speaker: ~150 words per minute = 0.4s per word
            # Add character-based component for longer words
            return max(0.5, (words * 0.4) + (chars * 0.01))
        
        duration = estimate_duration(text)
        
        # Add to transcript buffer for potential clip subtitles
        transcript_item = {
            "start": timestamp,
            "end": timestamp + duration,
            "text": text
        }
        transcript_buffer.append(transcript_item)
        
        # Keep only recent transcripts (last 60 seconds)
        current_time = time.time()
        while transcript_buffer and transcript_buffer[0]["start"] < current_time - 60:
            transcript_buffer.pop(0)
        
        # Process the transcription with each user's trigger detector
        for user_id, detector in trigger_detectors.items():
            detector.process_transcription(timestamp, text)
        
        # Use Kimi-Audio for emotion detection if available
        if kimi_audio_available and audio_processor:
            try:
                emotion_result = audio_processor.detect_emotional_content(text)
                if emotion_result["has_emotion"] and emotion_result["intensity"] > 0.5:
                    # Only store significant emotional content
                    emotion_buffer.append({
                        "timestamp": timestamp,
                        "text": text,
                        "emotion": emotion_result
                    })
                    
                    # If it's a strong positive or surprise emotion, trigger a clip
                    if (emotion_result["emotion_type"] in ["positive", "positive_surprise"] 
                            and emotion_result["intensity"] > 0.7):
                        for user_id in users:
                            # Create an emotional trigger event
                            event = TriggerEvent(
                                timestamp=timestamp,
                                reason=f"emotion:{emotion_result['emotion_type']}",
                                metadata={
                                    "emotion_type": emotion_result["emotion_type"],
                                    "emotion_words": emotion_result["emotion_words"],
                                    "intensity": emotion_result["intensity"]
                                }
                            )
                            on_trigger(event, user_id, streamer_id)
            except Exception as e:
                logger.error(f"Error in emotion detection: {str(e)}")
    
    stt_service = RTSTTService(on_transcription)
    stt_service.start()
    
    # Step 9: Set up shared stream listener with improved buffering
    def on_trigger(event: TriggerEvent, user_id: str, streamer_id: str):
        """Handle trigger events from any source."""
        logger.info(f"Trigger for {user_id}: {event.reason} at {event.timestamp:.2f}s")
        
        # Generate clip paths using the clips organizer
        video_path, metadata_path = clips_organizer.get_clip_path(
            user_id=user_id,
            streamer_id=streamer_id,
            timestamp=event.timestamp
        )
        
        # Calculate clip boundaries
        clip_start = max(0, event.timestamp - 15.0)  # 15 seconds before trigger
        clip_end = event.timestamp + 15.0  # 15 seconds after trigger
        
        # For Kimi-Audio triggered events, use the clip suggester to refine boundaries
        if (kimi_audio_available and clip_suggester and 
                event.reason.startswith("emotion:") and os.path.exists(video_file)):
            try:
                logger.info("Using Kimi-Audio to suggest optimal clip boundaries")
                suggestions = clip_suggester.suggest_clips(
                    media_path=video_file,
                    center_timestamp=event.timestamp,
                    keywords=user_configs[user_id]["keywords"],
                    max_suggestions=1
                )
                
                if suggestions and len(suggestions) > 0:
                    suggestion = suggestions[0]
                    # Use suggested boundaries if available
                    clip_start = suggestion["start"]
                    clip_end = suggestion["end"]
                    logger.info(f"Refined clip boundaries: {clip_start:.2f}s - {clip_end:.2f}s")
            except Exception as e:
                logger.error(f"Error using clip suggester: {str(e)}")
                logger.warning("Using default clip boundaries")
        
        # If preference model is available, use it to personalize clip selection
        if learning_available and preference_model:
            # Adjust clip boundaries based on user preferences
            try:
                if event.reason.startswith("keyword:"):
                    keyword = event.metadata.get("keyword", "")
                    if keyword:
                        # Check keyword preference
                        kw_pref = preference_model.get_keyword_preference(user_id, keyword)
                        
                        # For highly preferred keywords, make slightly longer clips
                        if kw_pref > 0.7:
                            post_padding_bonus = 5.0  # Add 5 more seconds for preferred keywords
                            clip_end += post_padding_bonus
                            logger.info(f"Extended clip for preferred keyword '{keyword}' (preference: {kw_pref:.2f})")
                        
                # For emotional content, check emotion preference
                elif event.reason.startswith("emotion:"):
                    emotion_type = event.metadata.get("emotion_type", "")
                    if emotion_type:
                        emotion_pref = preference_model.get_emotion_preference(user_id, emotion_type)
                        
                        # For highly preferred emotions, make slightly longer clips
                        if emotion_pref > 0.7:
                            post_padding_bonus = 5.0
                            clip_end += post_padding_bonus
                            logger.info(f"Extended clip for preferred emotion '{emotion_type}' (preference: {emotion_pref:.2f})")
            except Exception as e:
                logger.error(f"Error applying preferences: {str(e)}")
        
        try:
            # Extract recent transcript for the clip
            recent_transcript = [
                t for t in transcript_buffer
                if t["start"] >= clip_start and t["end"] <= clip_end
            ]
            
            # Create clip metadata
            clip_metadata = {
                "user_id": user_id, 
                "trigger_reason": event.reason,
                **event.metadata
            }
            
            if event.reason.startswith("keyword:"):
                clip_metadata["keywords"] = [event.metadata.get("keyword", "")]
            
            # Create the clip using ClipService
            clip_service.create_clip(
                video_path=video_file,
                center_ts=event.timestamp,
                reason=event.reason,
                subtitles=[(t["start"], t["end"], t["text"]) for t in recent_transcript],
                metadata=clip_metadata,
                output_path=video_path,
                metadata_path=metadata_path
            )
            
            # Generate additional metadata using Kimi-Audio if available
            additional_metadata = {}
            if kimi_audio_available and audio_processor:
                try:
                    # Generate a concise caption for the clip
                    temp_clip_file = clip_service.get_last_clip_path()
                    if os.path.exists(temp_clip_file):
                        caption = audio_processor.generate_audio_caption(temp_clip_file)
                        additional_metadata["audio_caption"] = caption
                        logger.info(f"Generated audio caption: {caption}")
                        
                        # Collect audio sample for model fine-tuning if enabled
                        if learning_available and model_finetuner:
                            # This would capture audio for future fine-tuning
                            # In a real implementation, we would extract actual audio features
                            # Here we just log that it would happen
                            logger.info(f"Would collect audio sample for user {user_id} (fine-tuning)")
                except Exception as e:
                    logger.error(f"Error generating audio caption: {str(e)}")
            
            # Generate and save metadata using ClipsOrganizer
            metadata = clips_organizer.generate_metadata(
                user_id=user_id,
                streamer_id=streamer_id,
                trigger_time=event.timestamp,
                clip_start=clip_start,
                clip_end=clip_end,
                trigger_reason=event.reason,
                transcript=recent_transcript,
                **additional_metadata
            )
            
            clips_organizer.save_clip_metadata(metadata_path, metadata)
            
            logger.info(f"Created clip for {user_id} at {video_path}")
            
            # Simulate user feedback for demo purposes
            # In a real application, this would come from UI interactions
            if learning_available:
                # Choose a random feedback option
                feedback_options = ["keep", "favorite", "share", "discard", "skip"]
                # Bias toward positive feedback for demo purposes
                weights = [0.4, 0.2, 0.1, 0.2, 0.1]
                
                import random
                feedback = random.choices(feedback_options, weights=weights)[0]
                
                # Process the feedback
                clip_id = os.path.basename(video_path)
                process_clip_feedback(
                    clip_id=clip_id,
                    user_id=user_id,
                    feedback=feedback,
                    trigger_reason=event.reason,
                    metadata=metadata
                )
                
                logger.info(f"Simulated user feedback for clip: {feedback}")
        except Exception as e:
            logger.error(f"Error creating clip: {str(e)}")
    
    # Start the chat service
    chat_service.start()
    
    try:
        # Start processing the video
        logger.info(f"Processing video: {video_file}")
        logger.info("Press Ctrl+C to stop")
        
        # Create and start the shared stream listener
        listener = SharedStreamListener(video_file, push_audio=stt_service.push)
        listener.start()
        
        # Let it run for a while to generate clips
        total_runtime = 30  # Run for 30 seconds
        logger.info(f"Running for {total_runtime} seconds...")
        
        # Sleep in small increments to allow keyboard interrupts
        for _ in range(total_runtime):
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Stopping processing")
    finally:
        # Clean up
        logger.info("Shutting down services...")
        chat_service.stop()
        if hasattr(stt_service, 'shutdown'):
            stt_service.shutdown()
        
        # Give some time for everything to finish
        time.sleep(1)
        
        # Print summary of clips generated
        clip_counts = clips_organizer.get_clip_count()
        logger.info("Clip generation summary:")
        for user_id, count in clip_counts.items():
            logger.info(f"  {user_id}: {count} clips")
        
        # List any database files created by learning module
        logger.info("Learning database files:")
        if os.path.exists("db"):
            for file in os.listdir("db"):
                logger.info(f"  db/{file}")

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

if __name__ == "__main__":
    import sys
    import torch  # Import for cuda availability check
    
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <video_file>")
        sys.exit(1)
    
    main(sys.argv[1]) 