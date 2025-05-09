"""
Simplified Speech-to-Text service for testing.
"""

import logging
import time
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)

class RTSTTService:
    """
    Simplified Speech-to-Text Service for testing.
    """
    
    def __init__(self, callback: Callable[[str, Optional[float]], None]):
        """
        Initialize the service.
        
        Args:
            callback: Function to call with transcription results
        """
        self.callback = callback
        self.running = False
        self.thread = None
        
        # Sample transcriptions for testing with trigger words
        self.test_transcriptions = [
            "This is a test transcription",
            "Wow this is amazing content",
            "Cool feature that works really well",
            "Let's try something interesting today",
            "This is awesome functionality",
            "Nice job implementing this",
            "Let's go, this is amazing",
            "Wow I can't believe how awesome this is",
            "That's really cool how it works",
            "Interesting results from the test"
        ]
        
        logger.info("Initialized RTSTTService with sample transcriptions")
        
    def start(self):
        """Start the service."""
        self.running = True
        self.thread = threading.Thread(target=self._send_sample_transcriptions)
        self.thread.daemon = True
        self.thread.start()
        logger.info("Started RTSTTService")
    
    def shutdown(self):
        """Shut down the service."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        logger.info("Stopped RTSTTService")
    
    def push(self, audio_data, **kwargs):
        """
        Process audio data (mock implementation).
        
        Args:
            audio_data: Audio data to process
            **kwargs: Additional arguments
        """
        # We don't actually process the audio in this mock version
        pass
    
    def _send_sample_transcriptions(self):
        """Send sample transcriptions at intervals."""
        idx = 0
        while self.running:
            if idx < len(self.test_transcriptions):
                text = self.test_transcriptions[idx]
                timestamp = time.time()
                self.callback(text, timestamp)
                idx += 1
            else:
                # Loop back to beginning
                idx = 0
            
            # Sleep between transcriptions
            time.sleep(2.0)
