"""
Live stream recording service for MX Clipping.

This module handles recording from live streams using ffmpeg,
with automatic platform URL resolution.
"""

import os
import time
import subprocess
import logging
import threading
import datetime
from typing import Optional, Dict, Any, Callable

from .stream_resolver import StreamResolver

logger = logging.getLogger(__name__)

class LiveRecordingService:
    """
    Service for recording live streams from various platforms.
    
    Features:
    - Automatic platform URL resolution (Twitch, YouTube, etc.)
    - Background recording process
    - Segment-based recording for reliability
    - Callback on completion or error
    """
    
    def __init__(
        self,
        user_url: str,
        output_dir: str = "recordings",
        segment_time: int = 300,  # 5 minute segments
        quality: str = "best",
        max_duration: Optional[int] = None,
        completion_callback: Optional[Callable[[str, bool, Optional[str]], None]] = None
    ):
        """
        Initialize the live recording service.
        
        Args:
            user_url: Platform URL or direct stream URL
            output_dir: Directory to store recordings
            segment_time: Length of each recording segment in seconds
            quality: Stream quality to record (best, worst, 720p, etc.)
            max_duration: Maximum recording duration in seconds (None for unlimited)
            completion_callback: Function to call when recording completes or fails
        """
        self.user_url = user_url
        self.output_dir = output_dir
        self.segment_time = segment_time
        self.quality = quality
        self.max_duration = max_duration
        self.completion_callback = completion_callback
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize stream resolver
        self.resolver = StreamResolver()
        
        # Recording state
        self.process = None
        self.recording_thread = None
        self.stop_flag = threading.Event()
        self.output_file = None
        self.resolved_url = None
        self.platform = None
        self.error = None
        
    def start(self) -> bool:
        """
        Start recording the live stream.
        
        Returns:
            True if recording started successfully, False otherwise
        """
        # Resolve the stream URL
        self.resolved_url, error = self.resolver.resolve_stream_url(self.user_url, self.quality)
        
        if not self.resolved_url:
            logger.error(f"Failed to resolve stream URL: {error}")
            self.error = f"Stream URL resolution failed: {error}"
            if self.completion_callback:
                self.completion_callback(self.user_url, False, self.error)
            return False
        
        # Detect platform for file naming
        self.platform = self.resolver.detect_platform(self.user_url)
        
        # Set up output filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        streamer = self._extract_streamer_name(self.user_url, self.platform)
        self.output_file = os.path.join(
            self.output_dir,
            f"{self.platform}_{streamer}_{timestamp}"
        )
        
        # Start recording in a separate thread
        self.recording_thread = threading.Thread(
            target=self._record_stream,
            daemon=True
        )
        self.recording_thread.start()
        
        logger.info(f"Started recording {self.user_url} to {self.output_file}")
        return True
    
    def stop(self) -> None:
        """Stop the recording process."""
        if not self.recording_thread:
            logger.warning("No active recording to stop")
            return
        
        logger.info("Stopping recording...")
        self.stop_flag.set()
        
        # Wait for recording thread to finish (with timeout)
        if self.recording_thread.is_alive():
            self.recording_thread.join(timeout=10.0)
        
        # Force kill the process if it's still running
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
                time.sleep(2)
                if self.process.poll() is None:
                    self.process.kill()
            except Exception as e:
                logger.error(f"Error terminating process: {str(e)}")
    
    def _record_stream(self) -> None:
        """Internal method to handle the recording process."""
        start_time = time.time()
        
        try:
            # Set up ffmpeg command with segment options
            segment_pattern = f"{self.output_file}_%03d.mp4"
            
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "warning",
                "-i", self.resolved_url,
                "-c", "copy",                  # Copy streams without re-encoding
                "-f", "segment",               # Enable segmented output
                "-segment_time", str(self.segment_time),  # Segment duration
                "-reset_timestamps", "1",      # Reset timestamps for each segment
                "-segment_format", "mp4",      # Output format
                segment_pattern
            ]
            
            # Start ffmpeg process
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            logger.info(f"FFmpeg recording process started (PID: {self.process.pid})")
            
            # Monitor the process
            while self.process.poll() is None:
                # Check if we need to stop
                if self.stop_flag.is_set():
                    logger.info("Stop requested, terminating recording")
                    self.process.terminate()
                    break
                
                # Check if we've exceeded max duration
                if self.max_duration and (time.time() - start_time) > self.max_duration:
                    logger.info(f"Maximum recording duration reached ({self.max_duration}s)")
                    self.process.terminate()
                    break
                
                # Sleep briefly to avoid busy-waiting
                time.sleep(1)
            
            # Process completed or was terminated
            exit_code = self.process.poll()
            
            # Get any error output
            stderr = self.process.stderr.read() if self.process.stderr else ""
            
            if exit_code != 0 and exit_code != -15:  # -15 is SIGTERM, which we use to stop
                self.error = f"FFmpeg exited with code {exit_code}: {stderr}"
                logger.error(self.error)
                
                if self.completion_callback:
                    self.completion_callback(self.output_file, False, self.error)
            else:
                logger.info(f"Recording completed: {self.output_file}")
                
                if self.completion_callback:
                    self.completion_callback(self.output_file, True, None)
        
        except Exception as e:
            self.error = f"Recording error: {str(e)}"
            logger.error(self.error, exc_info=True)
            
            if self.completion_callback:
                self.completion_callback(self.output_file, False, self.error)
    
    def _extract_streamer_name(self, url: str, platform: str) -> str:
        """
        Extract the streamer name from the URL.
        
        Args:
            url: The platform URL
            platform: The detected platform
            
        Returns:
            Streamer name or generic identifier
        """
        try:
            if platform == 'twitch':
                # Extract username from Twitch URL patterns
                import re
                match = re.search(r'twitch\.tv/([^/?&]+)', url)
                if match:
                    return match.group(1)
            
            elif platform == 'youtube':
                # For YouTube we'd need to extract channel information
                # This is simplified; more robust handling would be needed
                import re
                match = re.search(r'youtube\.com/(@[^/?&]+|c/[^/?&]+|channel/[^/?&]+)', url)
                if match:
                    return match.group(1).replace('/', '_')
                
            # For other platforms or if extraction fails, use a generic name
            return 'streamer'
        
        except Exception as e:
            logger.warning(f"Error extracting streamer name: {str(e)}")
            return 'streamer' 