"""
Mock Chat Service for testing.
"""

import threading
import time
import logging
import random
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)

class MockChatService:
    """
    Mock implementation of a chat service for testing.
    """
    
    def __init__(self, message_interval: float = 1.0):
        """
        Initialize the mock chat service.
        
        Args:
            message_interval: Time between messages in seconds
        """
        self.message_interval = message_interval
        self.running = False
        self.thread = None
        self.callback = None
        
        # Sample chat data for testing
        self.sample_usernames = ["user1", "user2", "streamer", "fan123", "viewer456"]
        self.sample_messages = [
            "Hello everyone!",
            "This stream is awesome",
            "Wow that was amazing",
            "Cool feature",
            "Nice work on this",
            "Interesting demo",
            "Let's go!",
            "How does this work?",
            "I like this",
            "Can you do that again?"
        ]
        
        logger.info("Initialized MockChatService")
    
    def set_callback(self, callback: Callable[[float, str, str], None]) -> None:
        """
        Set the callback function for new chat messages.
        
        Args:
            callback: Function to call with (timestamp, message, username)
        """
        self.callback = callback
    
    def start(self) -> None:
        """Start generating mock chat messages."""
        if self.thread and self.running:
            logger.warning("Chat service already running")
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._generate_messages)
        self.thread.daemon = True
        self.thread.start()
        
        logger.info("Started mock chat service")
    
    def stop(self) -> None:
        """Stop generating mock chat messages."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        
        logger.info("Stopped mock chat service")
    
    def _generate_messages(self) -> None:
        """Generate random chat messages at regular intervals."""
        if not self.callback:
            logger.warning("No callback set for chat messages")
            return
            
        while self.running:
            # Generate a random chat message
            username = random.choice(self.sample_usernames)
            message = random.choice(self.sample_messages)
            timestamp = time.time()
            
            # Call the callback
            try:
                self.callback(timestamp, message, username)
            except Exception as e:
                logger.error(f"Error in chat callback: {str(e)}")
            
            # Wait for the next message
            time.sleep(self.message_interval)


class ChatTrigger:
    """
    Monitors chat messages and triggers when message frequency exceeds a threshold.
    
    Uses a sliding window to calculate the mean and standard deviation of message 
    frequency, and triggers when the current frequency exceeds mean + threshold * std_dev.
    """
    
    def __init__(self, window_size: float = 5.0, threshold: float = 2.0, callback: Optional[Callable[[float, str], None]] = None):
        """
        Initialize the chat trigger.
        
        Args:
            window_size: The size of the sliding window in seconds
            threshold: The number of standard deviations above the mean to trigger
            callback: A function to call when a trigger occurs, takes (timestamp, reason) as arguments
        """
        self.window_size = window_size
        self.threshold = threshold
        self.callback = callback
        self.message_times = deque()
        self.lock = threading.Lock()
    
    def process_message(self, timestamp: float, message: str, username: str):
        """
        Process a new chat message and check if it triggers an event.
        
        Args:
            timestamp: The timestamp of the message
            message: The content of the message
            username: The username of the sender
        """
        with self.lock:
            # Add the current message time
            self.message_times.append(timestamp)
            
            # Remove messages outside the window
            cutoff_time = timestamp - self.window_size
            while self.message_times and self.message_times[0] < cutoff_time:
                self.message_times.popleft()
            
            # Count messages in window
            count = len(self.message_times)
            
            # If we have enough messages, check if we should trigger
            if count >= 3:  # Need some messages to calculate statistics
                # Calculate message frequency over time
                if len(self.message_times) >= 2:
                    intervals = [self.message_times[i] - self.message_times[i-1] 
                                for i in range(1, len(self.message_times))]
                    
                    if intervals:
                        mean_interval = np.mean(intervals)
                        std_interval = np.std(intervals)
                        
                        # If deviation is low, std might be close to 0, avoid division by zero
                        if std_interval > 0:
                            # Calculate current message frequency (messages per second)
                            frequency = count / self.window_size
                            # Expected frequency based on mean interval
                            expected_frequency = 1 / mean_interval if mean_interval > 0 else 0
                            
                            # If current frequency exceeds expected + threshold * std
                            if frequency > expected_frequency + self.threshold * (expected_frequency / mean_interval * std_interval):
                                if self.callback:
                                    reason = f"Chat spike: {count} messages in {self.window_size:.1f}s window"
                                    self.callback(timestamp, reason)
                                return True
        
        return False
    
    def set_callback(self, callback: Callable[[float, str], None]):
        """
        Set the callback function to be called when a trigger occurs.
        
        Args:
            callback: A function that takes (timestamp, reason) as arguments
        """
        self.callback = callback 