"""
mxclip CLI

Usage examples:
# Record microphone for N seconds and transcribe
python -m mxclip.main record --duration 5

# Play a local video file and transcribe its audio in real time
python -m mxclip.main video --file sample.mp4

# Run video analysis with chat trigger and clip generation
python -m mxclip.main analyze --video sample.mp4 --chat-freq 0.2

# Use Kimi-Audio to suggest optimal clip points
python -m mxclip.main suggest --video sample.mp4 --keywords "highlight,amazing"

# Find emotional moments in a video
python -m mxclip.main emotions --video sample.mp4 --output-dir clips

# Record from a streaming platform URL
python -m mxclip.main stream --url https://twitch.tv/username
"""
import argparse
import logging
import time
import os
import signal
import sys
import json
from typing import Dict, Any, Optional, List

from .stt_service import STTService
from .realtime_stt_service import RTSTTService
from .shared_stream_listener import SharedStreamListener
from .chat_service import MockChatService, ChatTrigger
from .clip_service import ClipService
from .user_processor import UserProcessorManager, TriggerEvent
from .metrics import initialize_metrics, get_metrics_service
from .stream_resolver import StreamResolver
from .live_recording_service import LiveRecordingService

# Import Kimi-Audio modules if available
try:
    from .audio_processor import KimiAudioProcessor
    from .clip_suggestion import ClipSuggester
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


def _add_subcommands(parser: argparse.ArgumentParser) -> None:
    """Configure subcommands."""
    sub = parser.add_subparsers(dest="cmd", required=True)

    # microphone demo
    rec = sub.add_parser("record", help="Record microphone and transcribe")
    rec.add_argument("--duration", type=int, default=5, help="Recording length in seconds")

    # video demo
    vid = sub.add_parser("video", help="Play local video and transcribe audio")
    vid.add_argument("--file", required=True, help="Path to local mp4 / mkv / flv file")
    
    # full analysis demo
    analyze = sub.add_parser("analyze", help="Analyze video with chat triggers and generate clips")
    analyze.add_argument("--video", required=True, help="Path to local video file")
    analyze.add_argument("--output-dir", default="clips", help="Directory to store generated clips")
    analyze.add_argument("--chat-freq", type=float, default=0.2, 
                       help="Average time between chat messages in seconds")
    analyze.add_argument("--metrics-port", type=int, default=8000, 
                       help="Port to expose Prometheus metrics")
    
    # Kimi-Audio clip suggestions
    if KIMI_AUDIO_AVAILABLE:
        suggest = sub.add_parser("suggest", help="Use Kimi-Audio to suggest optimal clip points")
        suggest.add_argument("--video", required=True, help="Path to local video file")
        suggest.add_argument("--keywords", help="Comma-separated list of keywords to prioritize")
        suggest.add_argument("--output-dir", default="clips", help="Directory to store generated clips")
        suggest.add_argument("--max-suggestions", type=int, default=5, 
                           help="Maximum number of clip suggestions")
        suggest.add_argument("--create-clips", action="store_true", 
                           help="Automatically create clips from suggestions")
        suggest.add_argument("--min-duration", type=float, default=5.0,
                           help="Minimum clip duration in seconds")
        suggest.add_argument("--max-duration", type=float, default=60.0,
                           help="Maximum clip duration in seconds")
        
        # Emotion detection command
        emotions = sub.add_parser("emotions", help="Find emotional moments in video")
        emotions.add_argument("--video", required=True, help="Path to local video file")
        emotions.add_argument("--output-dir", default="clips", help="Directory to store generated clips")
        emotions.add_argument("--create-clips", action="store_true", help="Create clips of emotional moments")
        emotions.add_argument("--min-intensity", type=float, default=0.5, 
                            help="Minimum emotion intensity to detect (0.0-1.0)")
        emotions.add_argument("--types", default="all", 
                            help="Emotion types to detect (all, positive, negative, surprise, mixed)")
    
    # Stream recording
    stream = sub.add_parser("stream", help="Record from a streaming platform URL")
    stream.add_argument("--url", required=True, help="Platform URL (Twitch, YouTube, etc.)")
    stream.add_argument("--output-dir", default="recordings", help="Directory to store recordings")
    stream.add_argument("--segment-time", type=int, default=300, 
                      help="Length of each recording segment in seconds")
    stream.add_argument("--quality", default="best", help="Stream quality (best, worst, 720p, etc.)")
    stream.add_argument("--max-duration", type=int, default=None,
                      help="Maximum recording duration in seconds (None for unlimited)")


def _run_record(duration: int) -> None:
    """Run microphone recording demo."""
    stt = STTService()
    print(f"[mxclip] Recording for {duration} seconds… Speak now!")
    stt.demo_record(duration)


def _run_video(filepath: str) -> None:
    """Run video transcription demo."""
    def on_text(text: str) -> None:
        print("[STT]", text, flush=True)

    stt = RTSTTService(on_text)
    listener = SharedStreamListener(filepath, push_audio=stt.push)
    
    try:
        print(f"[mxclip] Transcribing audio from {filepath}...")
        listener.start()
    except KeyboardInterrupt:
        print("[mxclip] Stopping transcription...")
    finally:
        if hasattr(stt, 'shutdown'):
            stt.shutdown()


def _run_analyze(args) -> None:
    """Run full analysis with chat triggers and clip generation."""
    # Initialize services
    metrics = initialize_metrics(port=args.metrics_port)
    clip_service = ClipService(output_dir=args.output_dir)
    user_processor_manager = UserProcessorManager(
        clip_service=clip_service,
        video_path=args.video,
        clip_callback=_on_clip_created
    )
    
    # Set up signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print("\n[mxclip] Shutting down...")
        user_processor_manager.stop_all()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Set up chat service
    chat_service = MockChatService(message_interval=args.chat_freq)
    chat_trigger = ChatTrigger(window_size=5.0, threshold=2.0)
    
    # Set up speech-to-text service
    transcript_buffer = []
    
    def on_stt_text(text: str, actual_timestamp: Optional[float] = None) -> None:
        """Handle transcribed text.
        
        Args:
            text: The transcribed text
            actual_timestamp: Optional timestamp from the audio stream (if available)
        """
        # Use actual audio timestamp if available, otherwise current time
        timestamp = actual_timestamp if actual_timestamp is not None else time.time()
        print(f"[STT] {text}")
        
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
        transcript_buffer.append({
            "start": timestamp,
            "end": timestamp + duration,
            "text": text
        })
        
        # Keep only recent transcripts
        current_time = time.time()
        while transcript_buffer and transcript_buffer[0]["start"] < current_time - 60:
            transcript_buffer.pop(0)
    
    stt = RTSTTService(on_stt_text)
    
    # Connect chat service to chat trigger
    def on_chat_message(timestamp: float, message: str, username: str) -> None:
        """Handle incoming chat message."""
        print(f"[CHAT] {username}: {message}")
        
        # Record the trigger in metrics
        metrics.record_trigger(reason="chat", user_id=username)
        
        # Process message through trigger
        if chat_trigger.process_message(timestamp, message, username):
            # When triggered, create a trigger event with transcript
            trigger = TriggerEvent(
                timestamp=timestamp,
                reason="chat_spike",
                metadata={
                    "transcript": transcript_buffer.copy(),
                    "chat_messages": [{
                        "timestamp": timestamp,
                        "username": username,
                        "message": message
                    }]
                }
            )
            
            # Add trigger to user processor
            user_processor_manager.add_trigger(username, trigger)
    
    chat_service.set_callback(on_chat_message)
    
    # Connect chat trigger to user processor
    def on_chat_trigger(timestamp: float, reason: str) -> None:
        """Handle chat trigger events."""
        print(f"[TRIGGER] {reason} at {timestamp:.2f}s")
    
    chat_trigger.set_callback(on_chat_trigger)
    
    # Start processing
    chat_service.start()
    
    try:
        # Start audio processing
        listener = SharedStreamListener(args.video, push_audio=stt.push)
        print(f"[mxclip] Analyzing {args.video}...")
        print("[mxclip] Press Ctrl+C to stop")
        
        listener.start()
    except KeyboardInterrupt:
        pass
    finally:
        print("[mxclip] Shutting down...")
        chat_service.stop()
        user_processor_manager.stop_all()
        if hasattr(stt, 'shutdown'):
            stt.shutdown()


def _run_suggest(args) -> None:
    """Run Kimi-Audio clip suggestion."""
    if not KIMI_AUDIO_AVAILABLE:
        print("[ERROR] Kimi-Audio modules not available. Please install the required dependencies.")
        return
    
    # Parse keywords if provided
    keywords = []
    if args.keywords:
        keywords = [k.strip() for k in args.keywords.split(",")]
    
    print(f"[mxclip] Analyzing {args.video} for optimal clip points...")
    if keywords:
        print(f"[mxclip] Looking for keywords: {', '.join(keywords)}")
    
    try:
        # Initialize Kimi-Audio processor
        audio_processor = KimiAudioProcessor()
        
        # Initialize clip suggester
        clip_suggester = ClipSuggester(
            audio_processor=audio_processor,
            min_clip_duration=args.min_duration,
            max_clip_duration=args.max_duration
        )
        
        # Get clip suggestions
        suggestions = clip_suggester.suggest_clips(
            media_path=args.video,
            keywords=keywords,
            max_suggestions=args.max_suggestions
        )
        
        if not suggestions:
            print("[mxclip] No clip suggestions found.")
            return
        
        # Print suggestions
        print(f"\n[mxclip] Found {len(suggestions)} clip suggestions:")
        for i, suggestion in enumerate(suggestions):
            start = suggestion["start"]
            end = suggestion["end"]
            duration = end - start
            
            # Format times as HH:MM:SS
            def format_time(seconds):
                h = int(seconds // 3600)
                m = int((seconds % 3600) // 60)
                s = int(seconds % 60)
                return f"{h:02d}:{m:02d}:{s:02d}"
            
            start_str = format_time(start)
            end_str = format_time(end)
            
            print(f"\nSuggestion #{i+1}:")
            print(f"  Time: {start_str} - {end_str} (duration: {duration:.1f}s)")
            print(f"  Reason: {suggestion['reason']}")
            print(f"  Score: {suggestion['score']:.2f}")
            
            # Print text preview (truncated if too long)
            text = suggestion.get("text", "")
            if text:
                if len(text) > 100:
                    text = text[:97] + "..."
                print(f"  Content: \"{text}\"")
        
        # Save suggestions to JSON file
        output_file = os.path.join(args.output_dir, "clip_suggestions.json")
        os.makedirs(args.output_dir, exist_ok=True)
        
        with open(output_file, "w") as f:
            json.dump(suggestions, f, indent=2)
        
        print(f"\n[mxclip] Saved suggestions to {output_file}")
        
        # Create clips if requested
        if args.create_clips:
            print("\n[mxclip] Creating clips from suggestions...")
            
            # Initialize clip service
            clip_service = ClipService(output_dir=args.output_dir)
            
            # Create each suggested clip
            for i, suggestion in enumerate(suggestions):
                clip_name = f"suggested_clip_{i+1}"
                
                # Add metadata
                metadata = {
                    "suggestion": suggestion,
                    "source": args.video,
                    "keywords": keywords
                }
                
                # Create the clip
                clip_path = clip_service.create_clip(
                    video_path=args.video,
                    start_time=suggestion["start"],
                    end_time=suggestion["end"],
                    output_name=clip_name,
                    metadata=metadata
                )
                
                print(f"[mxclip] Created clip: {clip_path}")
    
    except Exception as e:
        print(f"[ERROR] {str(e)}")


def _run_emotions(args) -> None:
    """Find emotional moments in a video."""
    if not KIMI_AUDIO_AVAILABLE:
        print("[ERROR] Kimi-Audio modules not available. Please install the required dependencies.")
        return
    
    print(f"[mxclip] Analyzing {args.video} for emotional moments...")
    
    try:
        # Initialize Kimi-Audio processor
        audio_processor = KimiAudioProcessor()
        
        # Transcribe the video
        print("[mxclip] Transcribing video...")
        transcription = audio_processor.transcribe_audio(args.video)
        
        if "error" in transcription:
            print(f"[ERROR] Transcription failed: {transcription['error']}")
            return
        
        segments = transcription.get("segments", [])
        if not segments:
            print("[mxclip] No speech segments found in the video.")
            return
        
        # Analyze segments for emotional content
        print("[mxclip] Analyzing emotional content...")
        analyzed_segments = audio_processor.analyze_segments_for_emotion(segments)
        
        # Filter emotional segments by intensity and type
        emotional_segments = []
        for segment in analyzed_segments:
            if not segment["has_emotion"]:
                continue
                
            # Filter by intensity
            if segment["emotion_intensity"] < args.min_intensity:
                continue
                
            # Filter by emotion type
            if args.types != "all" and segment["emotion_type"]:
                emotion_type = segment["emotion_type"]
                if args.types not in emotion_type:
                    continue
            
            emotional_segments.append(segment)
        
        if not emotional_segments:
            print(f"[mxclip] No emotional moments found with intensity >= {args.min_intensity}.")
            return
        
        # Print emotional segments
        print(f"\n[mxclip] Found {len(emotional_segments)} emotional segments:")
        
        # Format times as HH:MM:SS
        def format_time(seconds):
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = int(seconds % 60)
            return f"{h:02d}:{m:02d}:{s:02d}"
        
        # Save results to JSON file
        results = []
        
        for i, segment in enumerate(emotional_segments):
            start = segment["start"]
            end = segment["end"]
            duration = end - start
            
            start_str = format_time(start)
            end_str = format_time(end)
            
            print(f"\nEmotion #{i+1}:")
            print(f"  Time: {start_str} - {end_str} (duration: {duration:.1f}s)")
            print(f"  Type: {segment['emotion_type']}")
            print(f"  Intensity: {segment['emotion_intensity']:.2f}")
            print(f"  Words: {', '.join(segment['emotion_words'])}")
            
            # Print text (truncated if too long)
            text = segment["text"]
            if len(text) > 100:
                text = text[:97] + "..."
            print(f"  Content: \"{text}\"")
            
            # Add to results
            results.append({
                "start": start,
                "end": end,
                "text": segment["text"],
                "emotion_type": segment["emotion_type"],
                "emotion_intensity": segment["emotion_intensity"],
                "emotion_words": segment["emotion_words"]
            })
        
        # Save results to JSON file
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir, "emotional_moments.json")
        
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n[mxclip] Saved emotional moments to {output_file}")
        
        # Create clips if requested
        if args.create_clips:
            print("\n[mxclip] Creating clips of emotional moments...")
            
            # Initialize clip service
            clip_service = ClipService(output_dir=args.output_dir)
            
            # Add some padding to emotional segments for better context
            PADDING_SEC = 2.0  # 2 seconds before and after
            
            # Create each emotional clip
            for i, segment in enumerate(results):
                # Add padding but don't go below 0
                start_time = max(0, segment["start"] - PADDING_SEC)
                end_time = segment["end"] + PADDING_SEC
                
                # Ensure minimum duration
                if end_time - start_time < 5.0:
                    end_time = start_time + 5.0
                
                emotion_type = segment["emotion_type"] or "emotion"
                clip_name = f"emotion_{emotion_type}_{i+1}"
                
                # Add metadata
                metadata = {
                    "emotion": segment,
                    "source": args.video
                }
                
                # Create the clip
                clip_path = clip_service.create_clip(
                    video_path=args.video,
                    start_time=start_time,
                    end_time=end_time,
                    output_name=clip_name,
                    metadata=metadata
                )
                
                print(f"[mxclip] Created clip: {clip_path}")
    
    except Exception as e:
        print(f"[ERROR] {str(e)}")


def _run_stream(args) -> None:
    """Record from a streaming platform URL."""
    print(f"[mxclip] Resolving stream URL: {args.url}")
    
    # Initialize the recording service
    def recording_complete(output_file, success, error_message):
        if success:
            print(f"[mxclip] Recording completed: {output_file}")
        else:
            print(f"[mxclip] Recording failed: {error_message}")
    
    recording_service = LiveRecordingService(
        user_url=args.url,
        output_dir=args.output_dir,
        segment_time=args.segment_time,
        quality=args.quality,
        max_duration=args.max_duration,
        completion_callback=recording_complete
    )
    
    # Start recording
    if recording_service.start():
        print(f"[mxclip] Recording started. Press Ctrl+C to stop.")
        
        try:
            # Keep running until interrupted
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[mxclip] Stopping recording...")
        finally:
            recording_service.stop()
            print("[mxclip] Recording stopped.")
    else:
        print(f"[ERROR] Failed to start recording: {recording_service.error}")


def _on_clip_created(clip_path: str, clip_info: Dict[str, Any]) -> None:
    """Handle clip creation event."""
    print(f"[CLIP] Created: {clip_path}")
    print(f"[CLIP] Reason: {clip_info.get('reason', 'unknown')}")
    print(f"[CLIP] User: {clip_info.get('user_id', 'unknown')}")
    
    # Record in metrics
    metrics = get_metrics_service()
    metrics.record_clip_created(
        reason=clip_info.get('reason', 'unknown'),
        user_id=clip_info.get('user_id', 'unknown'),
        processing_time=clip_info.get('processing_time', 0.0)
    )
    
    # For demonstration, mark all clips as relevant
    metrics.update_clip_precision(relevant=True)


def cli() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(prog="mxclip")
    _add_subcommands(parser)
    args = parser.parse_args()

    if args.cmd == "record":
        _run_record(args.duration)
    elif args.cmd == "video":
        _run_video(args.file)
    elif args.cmd == "analyze":
        _run_analyze(args)
    elif args.cmd == "suggest" and KIMI_AUDIO_AVAILABLE:
        _run_suggest(args)
    elif args.cmd == "emotions" and KIMI_AUDIO_AVAILABLE:
        _run_emotions(args)
    elif args.cmd == "stream":
        _run_stream(args)


if __name__ == "__main__":
    cli()
