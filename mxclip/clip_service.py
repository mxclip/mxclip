"""
Clip Service for MX Clipping.

This module handles creating video clips from source videos,
adding subtitles, watermarks, and generating metadata.
"""

import os
import logging
import json
import time
import tempfile
import ffmpeg
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)

class ClipService:
    """
    Service for creating clips from video files.
    """
    
    def __init__(self, 
                output_dir: str = "clips", 
                clip_length: float = 30.0,
                pre_padding: float = 15.0,
                post_padding: float = 15.0,
                watermark_path: Optional[str] = None,
                watermark_position: str = "bottomright",
                watermark_size: float = 0.2,
                max_duration: float = 60.0):
        """
        Initialize the clip service.
        
        Args:
            output_dir: Directory to store clips
            clip_length: Default clip length in seconds
            pre_padding: Seconds before trigger to include in clip
            post_padding: Seconds after trigger to include in clip
            watermark_path: Path to watermark image file
            watermark_position: Position of watermark (bottomright, bottomleft, etc.)
            watermark_size: Size of watermark as a fraction of video size
            max_duration: Maximum clip duration in seconds
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.clip_length = clip_length
        self.pre_padding = pre_padding
        self.post_padding = post_padding
        self.watermark_path = watermark_path
        self.watermark_position = watermark_position
        self.watermark_size = watermark_size
        self.max_duration = max_duration
        self.last_clip_path = None
        
        logger.info(f"Initialized ClipService with output directory: {output_dir}")
    
    def create_clip(
        self,
        video_path: str,
        center_ts: float,
        reason: str,
        output_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
        subtitles: Optional[List[Tuple[float, float, str]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        duration: Optional[float] = None,
    ) -> str:
        """
        Create a video clip from a specific point in the source video.
        
        Args:
            video_path: Path to the source video
            center_ts: Timestamp to center the clip on (seconds from start)
            reason: Reason for creating the clip
            output_path: Path to save the clip (or auto-generate if None)
            metadata_path: Path to save metadata (or auto-generate if None)
            subtitles: List of (start, end, text) for subtitles
            metadata: Additional metadata to include
            duration: Custom duration to override default clip_length
            
        Returns:
            Path to the created clip
        """
        # Get video information
        try:
            video_info = ffmpeg.probe(video_path)
            video_duration = float(video_info['format']['duration'])
        except Exception as e:
            logger.error(f"Error probing video: {str(e)}")
            video_duration = 3600  # Assume 1 hour if can't determine
        
        # Calculate clip boundaries
        clip_start = max(0, center_ts - self.pre_padding)
        clip_end = min(video_duration, center_ts + self.post_padding)
        
        # Enforce maximum duration
        clip_duration = clip_end - clip_start
        if clip_duration > self.max_duration:
            # Keep the center timestamp but reduce duration
            excess = clip_duration - self.max_duration
            clip_start += excess / 2
            clip_end -= excess / 2
            clip_duration = clip_end - clip_start
        
        # Generate output path if not provided
        if not output_path:
            timestamp = int(time.time())
            os.makedirs(self.output_dir, exist_ok=True)
            output_path = os.path.join(self.output_dir, f"clip_{timestamp}.mp4")
        
        # Generate metadata path if not provided
        if not metadata_path:
            metadata_path = output_path.replace(".mp4", ".json")
        
        # Create subtitle file if needed
        subtitle_path = None
        if subtitles and len(subtitles) > 0:
            subtitle_path = self._create_subtitle_file(subtitles, clip_start)
        
        try:
            # Set up ffmpeg command for clip extraction
            stream = ffmpeg.input(video_path, ss=clip_start, t=clip_duration)
            
            # Add watermark if provided
            if self.watermark_path and os.path.exists(self.watermark_path):
                stream = self._add_watermark(stream)
            
            # Add subtitles if provided
            if subtitle_path:
                stream = ffmpeg.output(
                    stream, 
                    output_path,
                    vf=f"subtitles={subtitle_path}",
                    c='copy',
                    avoid_negative_ts='make_zero'
                )
            else:
                stream = ffmpeg.output(
                    stream, 
                    output_path,
                    c='copy',
                    avoid_negative_ts='make_zero'
                )
            
            # In a real implementation, we would run ffmpeg
            # For the mock version, just create the file
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                f.write("# Mock video clip file")
            
            # Save metadata
            clip_info = {
                "source": video_path,
                "clip_start": clip_start,
                "clip_end": clip_end,
                "duration": clip_end - clip_start,
                "center_timestamp": center_ts,
                "reason": reason,
                "created_at": time.time(),
                "subtitles": subtitles or [],
                "output_path": output_path
            }
            
            # Add custom metadata if provided
            if metadata:
                clip_info.update(metadata)
            
            # Save metadata to file
            os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
            with open(metadata_path, 'w') as f:
                json.dump(clip_info, f, indent=2)
            
            # Store last clip path for reference
            self.last_clip_path = output_path
            
            logger.info(f"Created mock clip at {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating clip: {str(e)}")
            raise
    
    def _create_subtitle_file(self, subtitles: List[Tuple[float, float, str]], start_offset: float) -> str:
        """
        Create a subtitle file from the given subtitle data.
        
        Args:
            subtitles: List of (start_time, end_time, text) tuples
            start_offset: Time offset for the start of the clip
            
        Returns:
            Path to the created subtitle file
        """
        with tempfile.NamedTemporaryFile(suffix='.srt', delete=False) as temp_file:
            subtitle_path = temp_file.name
            
            # Write subtitles in SRT format
            for i, (start, end, text) in enumerate(subtitles, 1):
                # Adjust times to clip start
                adj_start = max(0, start - start_offset)
                adj_end = max(0, end - start_offset)
                
                # Only include subtitles that are within the clip
                if adj_end > 0:
                    # Convert to SRT time format (HH:MM:SS,mmm)
                    start_str = self._format_srt_time(adj_start)
                    end_str = self._format_srt_time(adj_end)
                    
                    # Write subtitle entry
                    temp_file.write(f"{i}\n".encode('utf-8'))
                    temp_file.write(f"{start_str} --> {end_str}\n".encode('utf-8'))
                    temp_file.write(f"{text}\n\n".encode('utf-8'))
            
            return subtitle_path
    
    def _format_srt_time(self, seconds: float) -> str:
        """
        Format seconds as SRT time string (HH:MM:SS,mmm).
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted time string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{int((seconds % 1) * 1000):03d}"
    
    def _add_watermark(self, stream):
        """
        Add watermark to video stream.
        
        Args:
            stream: FFmpeg video stream
            
        Returns:
            Modified stream with watermark
        """
        # Position mapping
        positions = {
            "bottomright": "W-w-10:H-h-10",
            "bottomleft": "10:H-h-10",
            "topright": "W-w-10:10",
            "topleft": "10:10",
            "center": "(W-w)/2:(H-h)/2"
        }
        position = positions.get(self.watermark_position, positions["bottomright"])
        
        # Add watermark overlay
        return ffmpeg.overlay(
            stream,
            ffmpeg.input(self.watermark_path).filter('scale', f'W*{self.watermark_size}', -1),
            x=position.split(':')[0],
            y=position.split(':')[1]
        )
    
    def get_last_clip_path(self) -> Optional[str]:
        """
        Get the path to the last created clip.
        
        Returns:
            Path to the last clip or None if no clips created
        """
        return self.last_clip_path 