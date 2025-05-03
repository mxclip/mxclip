import time
import random
import threading
import numpy as np
from collections import deque
from datetime import datetime, timedelta
from typing import Iterator, Tuple, Optional, List, Callable

class MockChatService:
    """
    A mock chat service that generates random messages for demo purposes.
    
    Yields:
        Tuple[float, str, str]: A tuple containing (timestamp, message, username)
    """
    SAMPLE_MESSAGES = [
        "Hello everyone!",
        "This gameplay is amazing!",
        "What a great play!",
        "OMG did you see that?",
        "LOL that was unexpected",
        "Wow, nice move!",
        "That's insane!",
        "Let's go!",
        "GG everyone",
        "That was close!"
    ]
    
    SAMPLE_USERNAMES = [
        "gamer123",
        "proStreamer",
        "coolPlayer42",
        "epicGamer",
        "gameEnthusiast",
        "playerOne",
        "streamViewer",
        "chatLover",
        "fanBoy123",
        "gamingFan"
    ]
    
    def __init__(self, message_interval: float = 0.2):
        """
        Initialize the mock chat service.
        
        Args:
            message_interval: The average time between messages in seconds
        """
        self.message_interval = message_interval
        self.running = False
        self.thread = None
    
    def start(self):
        """Start the mock chat service."""
        self.running = True
        self.thread = threading.Thread(target=self._generate_messages, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop the mock chat service."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
    
    def _generate_messages(self):
        """Generate random messages at random intervals."""
        while self.running:
            # Sleep for a random time around the message interval
            time.sleep(max(0.05, random.gauss(self.message_interval, self.message_interval / 4)))
            
            # Generate a random message
            message = random.choice(self.SAMPLE_MESSAGES)
            username = random.choice(self.SAMPLE_USERNAMES)
            timestamp = time.time()
            
            # Call the callback with the message
            if hasattr(self, 'callback') and callable(self.callback):
                self.callback(timestamp, message, username)
    
    def set_callback(self, callback: Callable[[float, str, str], None]):
        """
        Set a callback function to receive messages.
        
        Args:
            callback: A function that takes (timestamp, message, username) as arguments
        """
        self.callback = callback
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


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