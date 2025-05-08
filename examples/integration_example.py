"""
Integration example for MX Clipping components.

This example demonstrates how the TriggerDetector, ConfigLoader, ClipsOrganizer,
and Kimi-Audio components work together with the existing MX Clipping system.
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

# Import Kimi-Audio components with proper error handling
try:
    from mxclip.audio_processor import KimiAudioProcessor
    from mxclip.clip_suggestion import ClipSuggester
    KIMI_AUDIO_AVAILABLE = True
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Kimi-Audio modules not available. Some features will be disabled.")
    KIMI_AUDIO_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main(video_file: str):
    """
    Run an integrated example of MX Clipping with new components.
    
    Args:
        video_file: Path to video file to process
    """
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
    
    # Step 4: Initialize Kimi-Audio components if available
    audio_processor = None
    clip_suggester = None
    
    if KIMI_AUDIO_AVAILABLE:
        try:
            logger.info("Initializing Kimi-Audio processor...")
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
            KIMI_AUDIO_AVAILABLE = False
    
    # Step 5: Set up trigger detectors for each user
    trigger_detectors = {}
    
    for user_id, config in user_configs.items():
        # Create a trigger detector with config settings
        detector = TriggerDetector(
            keywords=config["keywords"],
            enable_repeat_check=config["enable_repeat_check"],
            repeat_window_seconds=config["repeat_window_seconds"],
            repeat_threshold=config["repeat_threshold"],
            enable_chat_check=config["enable_chat_check"],
            chat_activity_threshold=config["chat_activity_threshold"]
        )
        
        # Set up the callback for triggers
        detector.set_callback(lambda event, uid=user_id: on_trigger(event, uid, streamer_id))
        
        trigger_detectors[user_id] = detector
        logger.info(f"Created trigger detector for {user_id}")
    
    # Step 6: Set up the chat service
    chat_service = MockChatService(message_interval=0.5)
    
    # Process chat messages for each user's trigger detector
    def on_chat_message(timestamp: float, message: str, username: str):
        logger.info(f"Chat: {username}: {message}")
        for user_id, detector in trigger_detectors.items():
            detector.process_chat(timestamp, message, username)
    
    chat_service.set_callback(on_chat_message)
    
    # Step 7: Set up the STT service for transcription with improved buffering
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
        if KIMI_AUDIO_AVAILABLE and audio_processor:
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
    
    # Step 8: Set up shared stream listener with improved buffering
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
        if (KIMI_AUDIO_AVAILABLE and clip_suggester and 
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
        
        try:
            # Extract recent transcript for the clip
            recent_transcript = [
                t for t in transcript_buffer
                if t["start"] >= clip_start and t["end"] <= clip_end
            ]
            
            # Create the clip using ClipService
            clip_service.create_clip(
                video_path=video_file,
                center_ts=event.timestamp,
                reason=event.reason,
                subtitles=[(t["start"], t["end"], t["text"]) for t in recent_transcript],
                metadata={"user_id": user_id, **event.metadata},
                output_path=video_path,
                metadata_path=metadata_path
            )
            
            # Generate additional metadata using Kimi-Audio if available
            additional_metadata = {}
            if KIMI_AUDIO_AVAILABLE and audio_processor:
                try:
                    # Generate a concise caption for the clip
                    temp_clip_file = clip_service.get_last_clip_path()
                    if os.path.exists(temp_clip_file):
                        caption = audio_processor.generate_audio_caption(temp_clip_file)
                        additional_metadata["audio_caption"] = caption
                        logger.info(f"Generated audio caption: {caption}")
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
    except KeyboardInterrupt:
        logger.info("Stopping processing")
    finally:
        # Clean up
        chat_service.stop()
        if hasattr(stt_service, 'shutdown'):
            stt_service.shutdown()
        
        # Print summary of clips generated
        clip_counts = clips_organizer.get_clip_count()
        logger.info("Clip generation summary:")
        for user_id, count in clip_counts.items():
            logger.info(f"  {user_id}: {count} clips")

if __name__ == "__main__":
    import sys
    import torch  # Import for cuda availability check
    
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <video_file>")
        sys.exit(1)
    
    main(sys.argv[1]) 