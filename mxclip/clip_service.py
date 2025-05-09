"""
Mock Clip Service for testing.
"""

import os
import logging
import json
import time
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)

class ClipService:
    """
    Mock service for creating clips from video files.
    """
    
    def __init__(self, output_dir: str = "clips"):
        """
        Initialize the clip service.
        
        Args:
            output_dir: Directory to store clips
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
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
        clip_duration: float = 30.0,
    ) -> Dict[str, Any]:
        """
        Create a video clip (mock implementation).
        
        Args:
            video_path: Path to the source video
            center_ts: Timestamp to center the clip on (seconds from start)
            reason: Reason for creating the clip
            output_path: Path to save the clip (or auto-generate if None)
            metadata_path: Path to save metadata (or auto-generate if None)
            subtitles: List of (start, end, text) for subtitles
            metadata: Additional metadata to include
            clip_duration: Total duration of the clip in seconds
            
        Returns:
            Dictionary with clip information
        """
        # Calculate clip boundaries
        clip_start = max(0, center_ts - clip_duration/2)
        clip_end = center_ts + clip_duration/2
        
        # Generate output path if not provided
        if not output_path:
            timestamp = int(time.time())
            output_dir = os.path.join(self.output_dir, "clips")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"clip_{timestamp}.mp4")
        
        # Generate metadata path if not provided
        if not metadata_path:
            metadata_path = output_path.replace(".mp4", ".json")
        
        # In a real implementation, we would use ffmpeg to create the clip
        # For this mock version, we'll just create an empty file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write("# Mock video clip file")
        
        # Save metadata
        clip_info = {
            "source": video_path,
            "start_time": clip_start,
            "end_time": clip_end,
            "duration": clip_end - clip_start,
            "center_timestamp": center_ts,
            "reason": reason,
            "created_at": time.time(),
            "subtitles": subtitles or []
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
        return clip_info
    
    def get_last_clip_path(self) -> Optional[str]:
        """
        Get the path to the last created clip.
        
        Returns:
            Path to the last clip or None if no clips created
        """
        return self.last_clip_path 