"""
Integration example for MX Clipping components.

This example demonstrates how the TriggerDetector, ConfigLoader, and ClipsOrganizer
components work together with the existing MX Clipping system.
"""

import os
import time
import logging
import numpy as np
from typing import Dict, Any, List

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
    
    # Step 4: Set up trigger detectors for each user
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
    
    # Step 5: Set up the chat service
    chat_service = MockChatService(message_interval=0.5)
    
    # Process chat messages for each user's trigger detector
    def on_chat_message(timestamp: float, message: str, username: str):
        logger.info(f"Chat: {username}: {message}")
        for user_id, detector in trigger_detectors.items():
            detector.process_chat(timestamp, message, username)
    
    chat_service.set_callback(on_chat_message)
    
    # Step 6: Set up the STT service for transcription
    transcript_buffer = []
    
    def on_transcription(text: str):
        if not text:
            return
        
        timestamp = time.time()
        logger.info(f"Transcript: {text}")
        
        # Add to transcript buffer for potential clip subtitles
        transcript_buffer.append({
            "start": timestamp,
            "end": timestamp + len(text) * 0.1,  # Rough estimate
            "text": text
        })
        
        # Keep only recent transcripts (last 60 seconds)
        while transcript_buffer and transcript_buffer[0]["start"] < timestamp - 60:
            transcript_buffer.pop(0)
        
        # Process the transcription with each user's trigger detector
        for user_id, detector in trigger_detectors.items():
            detector.process_transcription(timestamp, text)
    
    stt_service = RTSTTService(on_transcription)
    
    # Step 7: Set up shared stream listener
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
        
        # Create the clip using the existing clip service
        # Note: In a real integration, we'd need to adapt ClipService to use our paths
        # Here, we're assuming it's been modified to accept custom paths
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
                metadata={"user_id": user_id, **event.metadata}
            )
            
            # Generate and save metadata using ClipsOrganizer
            metadata = clips_organizer.generate_metadata(
                user_id=user_id,
                streamer_id=streamer_id,
                trigger_time=event.timestamp,
                clip_start=clip_start,
                clip_end=clip_end,
                trigger_reason=event.reason,
                transcript=recent_transcript
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
    
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <video_file>")
        sys.exit(1)
    
    main(sys.argv[1]) 