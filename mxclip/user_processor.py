import threading
import logging
import time
import queue
from typing import Dict, Any, Callable, Optional, List, Tuple

from .clip_service import ClipService

logger = logging.getLogger(__name__)

class TriggerEvent:
    """
    Represents a trigger event for clip creation.
    """
    def __init__(self, timestamp: float, reason: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a trigger event.
        
        Args:
            timestamp: The timestamp of the event (in seconds)
            reason: The reason for the trigger
            metadata: Additional metadata related to the trigger
        """
        self.timestamp = timestamp
        self.reason = reason
        self.metadata = metadata or {}


class UserProcessor:
    """
    Processor that handles triggers for a specific user.
    
    Each user has their own processor instance to prevent cross-user clip mixing.
    The processor runs in a separate thread and processes trigger events to
    create clips using the clip service.
    """
    
    def __init__(
        self,
        user_id: str,
        clip_service: ClipService,
        video_path: str,
        min_interval: float = 5.0,
        max_queue_size: int = 100,
        callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
    ):
        """
        Initialize a user processor.
        
        Args:
            user_id: Unique identifier for the user
            clip_service: Service to create clips
            video_path: Path to the source video file
            min_interval: Minimum interval between clip creations in seconds
            max_queue_size: Maximum number of triggers to queue
            callback: Function to call when a clip is created
        """
        self.user_id = user_id
        self.clip_service = clip_service
        self.video_path = video_path
        self.min_interval = min_interval
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.callback = callback
        self.last_clip_time = 0
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        
        # Statistics
        self.triggers_received = 0
        self.clips_created = 0
        self.triggers_dropped = 0
        
    def start(self):
        """Start the user processor thread."""
        with self.lock:
            if not self.running:
                self.running = True
                self.thread = threading.Thread(target=self._process_loop, daemon=True)
                self.thread.start()
                logger.info(f"Started processor for user {self.user_id}")
    
    def stop(self):
        """Stop the user processor thread."""
        with self.lock:
            if self.running:
                self.running = False
                # Put a sentinel value to unblock the queue
                try:
                    self.queue.put(None, block=False)
                except queue.Full:
                    pass
                if self.thread:
                    self.thread.join(timeout=2.0)
                    logger.info(f"Stopped processor for user {self.user_id}")
    
    def add_trigger(self, trigger: TriggerEvent) -> bool:
        """
        Add a trigger event to the processing queue.
        
        Args:
            trigger: The trigger event to add
            
        Returns:
            True if the trigger was added, False otherwise
        """
        self.triggers_received += 1
        try:
            self.queue.put(trigger, block=False)
            return True
        except queue.Full:
            self.triggers_dropped += 1
            logger.warning(f"Trigger queue full for user {self.user_id}, dropping trigger")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the processor.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "user_id": self.user_id,
            "triggers_received": self.triggers_received,
            "clips_created": self.clips_created,
            "triggers_dropped": self.triggers_dropped,
            "queue_size": self.queue.qsize(),
            "running": self.running
        }
    
    def _process_loop(self):
        """Main processing loop that creates clips from triggers."""
        while self.running:
            try:
                # Get the next trigger from the queue
                trigger = self.queue.get(timeout=1.0)
                
                # Check for sentinel value indicating shutdown
                if trigger is None:
                    break
                
                # Check if enough time has passed since the last clip
                current_time = time.time()
                time_since_last = current_time - self.last_clip_time
                
                if time_since_last >= self.min_interval:
                    # Create subtitles from collected transcript if available
                    subtitles = None
                    if "transcript" in trigger.metadata:
                        transcript = trigger.metadata["transcript"]
                        if isinstance(transcript, list):
                            subtitles = [(item.get("start", 0), 
                                         item.get("end", 0), 
                                         item.get("text", "")) for item in transcript]
                        elif isinstance(transcript, str):
                            # Simple subtitle centered around the trigger time
                            subtitles = [(trigger.timestamp - 2, trigger.timestamp + 5, transcript)]
                    
                    # Create the clip
                    clip_path = self.clip_service.create_clip(
                        video_path=self.video_path,
                        center_ts=trigger.timestamp,
                        reason=trigger.reason,
                        subtitles=subtitles,
                        metadata={"user_id": self.user_id, **trigger.metadata}
                    )
                    
                    if clip_path:
                        self.clips_created += 1
                        self.last_clip_time = current_time
                        
                        # Call the callback if provided
                        if self.callback:
                            clip_info = {
                                "user_id": self.user_id,
                                "timestamp": trigger.timestamp,
                                "reason": trigger.reason,
                                "clip_path": clip_path,
                                **trigger.metadata
                            }
                            self.callback(clip_path, clip_info)
                        
                        logger.info(f"Created clip for user {self.user_id} at {trigger.timestamp:.2f}s")
                    else:
                        logger.error(f"Failed to create clip for user {self.user_id} at {trigger.timestamp:.2f}s")
                else:
                    logger.info(f"Skipping trigger for user {self.user_id}, too soon after last clip "
                                f"({time_since_last:.2f}s < {self.min_interval:.2f}s)")
            
            except queue.Empty:
                # Timeout on queue.get, just continue the loop
                pass
            except Exception as e:
                logger.error(f"Error processing trigger for user {self.user_id}: {str(e)}", exc_info=True)
        
        logger.info(f"Processor thread for user {self.user_id} exiting")


class UserProcessorManager:
    """
    Manages user processors for multiple users.
    
    This class creates and manages user processor instances for different users,
    ensuring that each user has their own independent processor.
    """
    
    def __init__(
        self,
        clip_service: ClipService,
        video_path: str,
        min_interval: float = 5.0,
        max_queue_size: int = 100,
        clip_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
    ):
        """
        Initialize the user processor manager.
        
        Args:
            clip_service: Service to create clips
            video_path: Path to the source video file
            min_interval: Minimum interval between clip creations in seconds
            max_queue_size: Maximum number of triggers to queue per user
            clip_callback: Function to call when a clip is created
        """
        self.clip_service = clip_service
        self.video_path = video_path
        self.min_interval = min_interval
        self.max_queue_size = max_queue_size
        self.clip_callback = clip_callback
        self.processors = {}
        self.lock = threading.Lock()
    
    def get_processor(self, user_id: str) -> UserProcessor:
        """
        Get or create a processor for the specified user.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            The user processor instance
        """
        with self.lock:
            if user_id not in self.processors:
                processor = UserProcessor(
                    user_id=user_id,
                    clip_service=self.clip_service,
                    video_path=self.video_path,
                    min_interval=self.min_interval,
                    max_queue_size=self.max_queue_size,
                    callback=self.clip_callback
                )
                processor.start()
                self.processors[user_id] = processor
            
            return self.processors[user_id]
    
    def add_trigger(self, user_id: str, trigger: TriggerEvent) -> bool:
        """
        Add a trigger event for a specific user.
        
        Args:
            user_id: Unique identifier for the user
            trigger: The trigger event to add
            
        Returns:
            True if the trigger was added, False otherwise
        """
        processor = self.get_processor(user_id)
        return processor.add_trigger(trigger)
    
    def stop_all(self):
        """Stop all user processors."""
        with self.lock:
            for processor in self.processors.values():
                processor.stop()
    
    def get_all_stats(self) -> List[Dict[str, Any]]:
        """
        Get statistics for all user processors.
        
        Returns:
            List of statistics dictionaries, one per processor
        """
        with self.lock:
            return [processor.get_stats() for processor in self.processors.values()] 