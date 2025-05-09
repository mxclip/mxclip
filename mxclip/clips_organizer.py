"""
Clips Organizer for MX Clipping.

This module handles the organization of generated clips,
including folder structure, naming conventions, and metadata management.
"""

import os
import json
import time
import logging
from datetime import datetime
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
    
    def get_clip_path(self, user_id: str, streamer_id: str, timestamp: float = None) -> Tuple[str, str]:
        """
        Generate paths for a new clip and its metadata.
        
        Args:
            user_id: User ID
            streamer_id: Streamer ID
            timestamp: Timestamp of the clip (defaults to current time if None)
            
        Returns:
            Tuple of (clip_path, metadata_path)
        """
        # Create directory structure
        user_dir = os.path.join(self.base_dir, user_id)
        streamer_dir = os.path.join(user_dir, streamer_id)
        os.makedirs(streamer_dir, exist_ok=True)
        
        # Generate filename based on timestamp or current time
        if timestamp is None:
            timestamp = time.time()
        
        # Format timestamp for filename with microseconds for uniqueness
        dt = datetime.fromtimestamp(timestamp)
        timestamp_str = dt.strftime("%Y%m%d_%H%M%S")
        unique_suffix = f"_{int(time.time() * 1000) % 10000}"  # Add milliseconds for uniqueness
        
        # Create filename with user, streamer and timestamp
        filename = f"{user_id}@{streamer_id}_{timestamp_str}{unique_suffix}"
        
        # Create paths
        clip_path = os.path.join(streamer_dir, f"{filename}.mp4")
        metadata_path = os.path.join(streamer_dir, f"{filename}.json")
        
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
            "user": user_id,
            "streamer": streamer_id,
            "trigger_time": trigger_time,
            "clip_start": clip_start,
            "clip_end": clip_end,
            "duration": clip_end - clip_start,
            "trigger_reason": trigger_reason,
            "created_at": datetime.now().isoformat()
        }
        
        # Add transcript if provided
        if transcript:
            metadata["transcript"] = transcript
        
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
            # Ensure directory exists
            os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Saved clip metadata to {metadata_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving clip metadata: {str(e)}")
            return False
    
    def list_user_clips(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all clips for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            List of clip metadata with additional filename, video_path and metadata_path
        """
        user_dir = os.path.join(self.base_dir, user_id)
        if not os.path.exists(user_dir):
            return []
        
        clips = []
        
        # Walk through user directory
        for root, _, files in os.walk(user_dir):
            for file in files:
                if file.endswith('.json'):
                    metadata_path = os.path.join(root, file)
                    video_path = metadata_path.replace('.json', '.mp4')
                    
                    # Skip if video file doesn't exist
                    if not os.path.exists(video_path):
                        continue
                    
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        
                        # Add paths to metadata
                        metadata["filename"] = os.path.basename(video_path)
                        metadata["video_path"] = video_path
                        metadata["metadata_path"] = metadata_path
                        
                        clips.append(metadata)
                    except Exception as e:
                        logger.error(f"Error loading clip metadata: {str(e)}")
        
        # Sort by creation time (newest first)
        clips.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        return clips
    
    def get_clip_count(self, user_id: str = None) -> Dict[str, int]:
        """
        Get the number of clips for each user or a specific user.
        
        Args:
            user_id: Optional user ID to get count for a specific user
            
        Returns:
            Dictionary mapping user IDs to clip counts,
            or a single count if user_id is specified
        """
        # Debug and display file structure
        self._print_file_structure()
        
        # If a specific user is requested, just count their clips
        if user_id:
            user_path = os.path.join(self.base_dir, user_id)
            if not os.path.exists(user_path):
                return 0
                
            count = 0
            # We need to recursively walk through all subdirectories
            for root, _, files in os.walk(user_path):
                for file in files:
                    if file.endswith('.mp4'):
                        count += 1
            
            self.clip_counts[user_id] = count
            return count
        
        # Otherwise update counts for all users
        self.clip_counts = {}  # Reset counts for a fresh scan
        
        if not os.path.exists(self.base_dir):
            return self.clip_counts
            
        for user_dir in os.listdir(self.base_dir):
            user_path = os.path.join(self.base_dir, user_dir)
            if os.path.isdir(user_path):
                count = 0
                for root, _, files in os.walk(user_path):
                    for file in files:
                        if file.endswith('.mp4'):
                            count += 1
                            
                self.clip_counts[user_dir] = count
        
        return self.clip_counts
        
    def _print_file_structure(self):
        """Print the file structure for debugging."""
        if not os.path.exists(self.base_dir):
            return
            
        print(f"\nDEBUG: File structure for {self.base_dir}:")
        for root, dirs, files in os.walk(self.base_dir):
            level = root.replace(self.base_dir, '').count(os.sep)
            indent = ' ' * 4 * level
            print(f"{indent}{os.path.basename(root)}/")
            sub_indent = ' ' * 4 * (level + 1)
            for f in files:
                print(f"{sub_indent}{f}")
    
    def get_user_clips(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Alias for list_user_clips for backward compatibility.
        
        Args:
            user_id: User ID
            
        Returns:
            List of clip metadata
        """
        return self.list_user_clips(user_id) 