"""
Clips Organizer for MX Clipping.

This module handles the organization of generated clips,
including folder structure, naming conventions, and metadata management.
"""

import os
import json
import time
import logging
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)

class ClipsOrganizer:
    """
    Organizes clips based on users and streamers.
    """
    
    def __init__(self, base_dir: str = "clips"):
        """
        Initialize the clips organizer.
        
        Args:
            base_dir: Base directory for clips
        """
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        
        # Track clip counts by user
        self.clip_counts = {}
        
        logger.info(f"Initialized ClipsOrganizer with base directory: {base_dir}")
    
    def get_clip_path(self, user_id: str, streamer_id: str, timestamp: float) -> Tuple[str, str]:
        """
        Generate paths for a new clip and its metadata.
        
        Args:
            user_id: User ID
            streamer_id: Streamer ID
            timestamp: Timestamp of the clip
            
        Returns:
            Tuple of (clip_path, metadata_path)
        """
        # Create directory structure
        user_dir = os.path.join(self.base_dir, user_id)
        streamer_dir = os.path.join(user_dir, streamer_id)
        os.makedirs(streamer_dir, exist_ok=True)
        
        # Generate filename
        timestamp_str = str(int(timestamp))
        clip_id = f"{timestamp_str}_{int(time.time())}"
        
        # Create paths
        clip_path = os.path.join(streamer_dir, f"{clip_id}.mp4")
        metadata_path = os.path.join(streamer_dir, f"{clip_id}.json")
        
        # Update clip count
        if user_id not in self.clip_counts:
            self.clip_counts[user_id] = 1
        else:
            self.clip_counts[user_id] += 1
        
        return clip_path, metadata_path
    
    def generate_metadata(self, user_id: str, streamer_id: str, trigger_time: float,
                        clip_start: float, clip_end: float, trigger_reason: str,
                        transcript: Optional[List[Dict[str, Any]]] = None,
                        **extra_metadata) -> Dict[str, Any]:
        """
        Generate metadata for a clip.
        
        Args:
            user_id: User ID
            streamer_id: Streamer ID
            trigger_time: Time when the trigger occurred
            clip_start: Start time of the clip
            clip_end: End time of the clip
            trigger_reason: Reason for the clip trigger
            transcript: List of transcript items
            **extra_metadata: Additional metadata
            
        Returns:
            Dictionary of metadata
        """
        metadata = {
            "user_id": user_id,
            "streamer_id": streamer_id,
            "trigger_time": trigger_time,
            "clip_start": clip_start,
            "clip_end": clip_end,
            "duration": clip_end - clip_start,
            "trigger_reason": trigger_reason,
            "created_at": time.time(),
            "transcript": transcript or []
        }
        
        # Add extra metadata
        metadata.update(extra_metadata)
        
        return metadata
    
    def save_clip_metadata(self, metadata_path: str, metadata: Dict[str, Any]) -> bool:
        """
        Save clip metadata to a file.
        
        Args:
            metadata_path: Path to save the metadata
            metadata: Metadata dictionary
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Saved clip metadata to {metadata_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving clip metadata: {str(e)}")
            return False
    
    def get_user_clips(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all clips for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            List of clip metadata
        """
        user_dir = os.path.join(self.base_dir, user_id)
        if not os.path.exists(user_dir):
            return []
        
        clips = []
        
        # Walk through user directory
        for root, _, files in os.walk(user_dir):
            for file in files:
                if file.endswith('.json'):
                    try:
                        with open(os.path.join(root, file), 'r') as f:
                            metadata = json.load(f)
                        clips.append(metadata)
                    except Exception as e:
                        logger.error(f"Error loading clip metadata: {str(e)}")
        
        # Sort by creation time (newest first)
        clips.sort(key=lambda x: x.get('created_at', 0), reverse=True)
        
        return clips
    
    def get_clip_count(self) -> Dict[str, int]:
        """
        Get the number of clips for each user.
        
        Returns:
            Dictionary mapping user IDs to clip counts
        """
        # Update counts by scanning directories
        for user_id in os.listdir(self.base_dir):
            user_dir = os.path.join(self.base_dir, user_id)
            if os.path.isdir(user_dir):
                count = 0
                for root, _, files in os.walk(user_dir):
                    for file in files:
                        if file.endswith('.mp4'):
                            count += 1
                self.clip_counts[user_id] = count
        
        return self.clip_counts 