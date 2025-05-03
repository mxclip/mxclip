"""
mxclip CLI

Usage examples:
# Record microphone for N seconds and transcribe
python -m mxclip.main record --duration 5

# Play a local video file and transcribe its audio in real time
python -m mxclip.main video --file sample.mp4

# Run video analysis with chat trigger and clip generation
python -m mxclip.main analyze --video sample.mp4 --chat-freq 0.2
"""
import argparse
import logging
import time
import os
import signal
import sys
from typing import Dict, Any, Optional

from .stt_service import STTService
from .realtime_stt_service import RTSTTService
from .shared_stream_listener import SharedStreamListener
from .chat_service import MockChatService, ChatTrigger
from .clip_service import ClipService
from .user_processor import UserProcessorManager, TriggerEvent
from .metrics import initialize_metrics, get_metrics_service

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
    
    def on_stt_text(text: str) -> None:
        """Handle transcribed text."""
        timestamp = time.time()
        print(f"[STT] {text}")
        
        # Add to transcript buffer for potential clip subtitles
        transcript_buffer.append({
            "start": timestamp,
            "end": timestamp + len(text) * 0.1,  # Rough estimate for subtitle duration
            "text": text
        })
        
        # Keep only recent transcripts
        while transcript_buffer and transcript_buffer[0]["start"] < timestamp - 60:
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


if __name__ == "__main__":
    cli()
