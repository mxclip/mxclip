import os
import json
import time
import subprocess
import tempfile
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
import ffmpeg

logger = logging.getLogger(__name__)

class ClipService:
    """
    Service for creating video clips with subtitles and watermarks.
    
    This service takes a timestamp and creates a video clip centered around that
    timestamp. It applies subtitles and watermarks to the clip and ensures that
    the output is no longer than a specified maximum duration.
    """
    
    def __init__(
        self, 
        output_dir: str = "clips",
        clip_length: float = 30.0,
        pre_padding: float = 15.0,
        post_padding: float = 15.0,
        watermark_path: Optional[str] = None,
        watermark_position: str = "bottomright",
        watermark_size: float = 0.2,
        max_duration: float = 30.0
    ):
        """
        Initialize the clip service.
        
        Args:
            output_dir: Directory to store the generated clips
            clip_length: Total desired clip length in seconds
            pre_padding: Amount of time to include before the center timestamp
            post_padding: Amount of time to include after the center timestamp
            watermark_path: Path to the watermark image
            watermark_position: Position of the watermark (topleft, topright, bottomleft, bottomright)
            watermark_size: Size of the watermark as a fraction of the video width
            max_duration: Maximum duration of the generated clip in seconds
        """
        self.output_dir = output_dir
        self.clip_length = clip_length
        self.pre_padding = pre_padding
        self.post_padding = post_padding
        self.watermark_path = watermark_path
        self.watermark_position = watermark_position
        self.watermark_size = watermark_size
        self.max_duration = max_duration
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def create_clip(
        self, 
        video_path: str,
        center_ts: float,
        reason: str,
        subtitles: Optional[List[Tuple[float, float, str]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Create a video clip centered around the specified timestamp.
        
        Args:
            video_path: Path to the source video file
            center_ts: Timestamp to center the clip around (in seconds)
            reason: Reason for creating the clip
            subtitles: List of subtitles as (start_time, end_time, text) tuples
            metadata: Additional metadata to include in the JSON file
        
        Returns:
            Path to the generated clip or None if an error occurred
        """
        start_time = time.time()
        clip_id = f"{int(center_ts)}_{int(time.time())}"
        output_path = os.path.join(self.output_dir, f"clip_{clip_id}.mp4")
        
        try:
            # Get video duration using ffprobe
            probe = ffmpeg.probe(video_path)
            duration = float(probe['format']['duration'])
            
            # Calculate clip start and end times
            start_time_sec = max(0, center_ts - self.pre_padding)
            end_time_sec = min(duration, center_ts + self.post_padding)
            
            # Ensure clip doesn't exceed max duration
            actual_duration = end_time_sec - start_time_sec
            if actual_duration > self.max_duration:
                # Trim clip to max duration while keeping it centered around center_ts
                excess = actual_duration - self.max_duration
                start_time_sec += excess / 2
                end_time_sec -= excess / 2
            
            # Run ffmpeg to generate the clip
            input_stream = ffmpeg.input(video_path, ss=start_time_sec, t=end_time_sec - start_time_sec)
            
            # Apply watermark if provided
            if self.watermark_path and os.path.exists(self.watermark_path):
                watermark = ffmpeg.input(self.watermark_path)
                
                # Set watermark position
                x_positions = {"left": "(w-overlay_w*{size})*0.05", "right": "(w-overlay_w*{size})*0.95"}
                y_positions = {"top": "(h-overlay_h*{size})*0.05", "bottom": "(h-overlay_h*{size})*0.95"}
                
                pos_parts = self.watermark_position.lower()
                vert = "top" if "top" in pos_parts else "bottom"
                horiz = "left" if "left" in pos_parts else "right"
                
                x_expr = x_positions[horiz].format(size=self.watermark_size)
                y_expr = y_positions[vert].format(size=self.watermark_size)
                
                overlay = ffmpeg.overlay(
                    input_stream, 
                    watermark.filter('scale', 
                                    f'iw*{self.watermark_size}', 
                                    f'ih*{self.watermark_size}'), 
                    x=x_expr,
                    y=y_expr
                )
            else:
                overlay = input_stream
            
            # Apply subtitles if provided
            if subtitles:
                # Create a temporary subtitle file in SRT format
                with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False) as srt:
                    for i, (sub_start, sub_end, text) in enumerate(subtitles, 1):
                        # Adjust subtitle times relative to clip start
                        rel_start = max(0, sub_start - start_time_sec)
                        rel_end = max(0, sub_end - start_time_sec)
                        
                        if rel_end > 0:  # Only include subtitles that appear in the clip
                            srt.write(f"{i}\n")
                            srt.write(f"{self._format_srt_time(rel_start)} --> {self._format_srt_time(rel_end)}\n")
                            srt.write(f"{text}\n\n")
                
                # Add subtitles to the video
                output = ffmpeg.output(
                    overlay,
                    output_path,
                    vf=f"subtitles={srt.name}:force_style='FontSize=24,Alignment=2,BorderStyle=3'",
                    c='copy'
                )
                
                # Clean up subtitle file after processing
                try:
                    output.run(quiet=True, overwrite_output=True)
                finally:
                    os.unlink(srt.name)
            else:
                # No subtitles, just output the video
                output = ffmpeg.output(overlay, output_path)
                output.run(quiet=True, overwrite_output=True)
            
            # Create JSON metadata file
            json_path = os.path.join(self.output_dir, f"clip_{clip_id}.json")
            json_data = {
                "clip_id": clip_id,
                "source": video_path,
                "center_timestamp": center_ts,
                "clip_start": start_time_sec,
                "clip_end": end_time_sec,
                "reason": reason,
                "created_at": datetime.now().isoformat(),
                "processing_time": time.time() - start_time,
            }
            
            if metadata:
                json_data.update(metadata)
            
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            logger.info(f"Clip successfully created: {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Error creating clip: {str(e)}", exc_info=True)
            # Clean up partial clip file if it exists
            if os.path.exists(output_path):
                try:
                    os.unlink(output_path)
                except:
                    pass
            return None
    
    def _format_srt_time(self, seconds: float) -> str:
        """
        Format seconds as SRT timestamp (HH:MM:SS,mmm).
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted SRT timestamp
        """
        ms = int((seconds - int(seconds)) * 1000)
        s = int(seconds % 60)
        m = int((seconds / 60) % 60)
        h = int(seconds / 3600)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}" 