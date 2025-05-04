"""
Clips organizer for MX Clipping.

This module handles the organization of generated clips,
including folder structure, naming conventions, and metadata management.
"""

import os
import json
import time
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

logger = logging.getLogger(__name__)

class ClipsOrganizer:
    """
    Organizes generated clips according to a consistent folder structure and naming convention.
    
    Format:
    clips/user1@streamer_20240502_231502_uniqueid.mp4
    clips/user1@streamer_20240502_231502_uniqueid.json
    """
    
    def __init__(self, base_dir: str = "clips"):
        """
        Initialize the clips organizer.
        
        Args:
            base_dir: Base directory for storing clips
        """
        self.base_dir = base_dir
        
        # Ensure the base directory exists
        os.makedirs(base_dir, exist_ok=True)
        
        # Create subdirectories for organization
        self.user_dirs = {}
    
    def get_clip_path(self, user_id: str, streamer_id: str, timestamp: Optional[float] = None) -> Tuple[str, str]:
        """
        Generate paths for a new clip and its metadata.
        
        Args:
            user_id: ID of the user who triggered the clip
            streamer_id: ID of the streamer being clipped
            timestamp: Optional timestamp for the clip (defaults to current time)
        
        Returns:
            Tuple of (video_path, metadata_path)
        """
        # Use current time if timestamp is not provided
        if timestamp is None:
            timestamp = time.time()
        
        # Format timestamp as YYYYMMDD_HHMMSS
        dt = datetime.fromtimestamp(timestamp)
        timestamp_str = dt.strftime("%Y%m%d_%H%M%S")
        
        # Generate a unique identifier to prevent collisions
        unique_id = str(uuid.uuid4())[:8]
        
        # Create filename: user@streamer_timestamp_uniqueid
        filename_base = f"{user_id}@{streamer_id}_{timestamp_str}_{unique_id}"
        
        # Create user directory if it doesn't exist
        user_dir = os.path.join(self.base_dir, user_id)
        if user_dir not in self.user_dirs:
            os.makedirs(user_dir, exist_ok=True)
            self.user_dirs[user_dir] = True
        
        # Create file paths
        video_path = os.path.join(user_dir, f"{filename_base}.mp4")
        metadata_path = os.path.join(user_dir, f"{filename_base}.json")
        
        return video_path, metadata_path
    
    def save_clip_metadata(self, metadata_path: str, metadata: Dict[str, Any]) -> None:
        """
        Save clip metadata to a JSON file.
        
        Args:
            metadata_path: Path to save the metadata file
            metadata: Metadata to save
        """
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Saved clip metadata to {metadata_path}")
        except Exception as e:
            logger.error(f"Error saving clip metadata: {str(e)}")
    
    def generate_metadata(
        self,
        user_id: str,
        streamer_id: str,
        trigger_time: float,
        clip_start: float,
        clip_end: float,
        trigger_reason: str,
        transcript: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Generate metadata for a clip.
        
        Args:
            user_id: ID of the user who triggered the clip
            streamer_id: ID of the streamer being clipped
            trigger_time: Timestamp when the trigger occurred
            clip_start: Start timestamp of the clip
            clip_end: End timestamp of the clip
            trigger_reason: Reason for the trigger
            transcript: Optional list of transcript segments
        
        Returns:
            Metadata dictionary
        """
        metadata = {
            "user": user_id,
            "streamer": streamer_id,
            "trigger_time": trigger_time,
            "clip_start": clip_start,
            "clip_end": clip_end,
            "trigger_reason": trigger_reason,
            "created_at": datetime.now().isoformat(),
        }
        
        if transcript:
            metadata["transcript"] = transcript
        
        return metadata
    
    def list_user_clips(self, user_id: str) -> List[Dict[str, Any]]:
        """
        List all clips for a specific user.
        
        Args:
            user_id: User ID to list clips for
        
        Returns:
            List of metadata dictionaries for the user's clips
        """
        clips = []
        user_dir = os.path.join(self.base_dir, user_id)
        
        if not os.path.exists(user_dir):
            return clips
        
        # Find all JSON metadata files in the user's directory
        for filename in os.listdir(user_dir):
            if filename.endswith(".json"):
                metadata_path = os.path.join(user_dir, filename)
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Add the filename to the metadata
                    metadata["filename"] = filename
                    metadata["video_path"] = os.path.join(user_dir, filename.replace(".json", ".mp4"))
                    metadata["metadata_path"] = metadata_path
                    
                    clips.append(metadata)
                except Exception as e:
                    logger.error(f"Error loading clip metadata from {metadata_path}: {str(e)}")
        
        # Sort clips by creation time (newest first)
        clips.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return clips
    
    def get_clip_count(self) -> Dict[str, int]:
        """
        Get the count of clips for each user.
        
        Returns:
            Dictionary mapping user IDs to clip counts
        """
        counts = {}
        
        for item in os.listdir(self.base_dir):
            user_dir = os.path.join(self.base_dir, item)
            if os.path.isdir(user_dir):
                # Count MP4 files in the user's directory
                user_id = item
                mp4_count = sum(1 for f in os.listdir(user_dir) if f.endswith(".mp4"))
                counts[user_id] = mp4_count
        
        return counts 