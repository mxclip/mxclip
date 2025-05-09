"""
Mock stream listener for testing.
"""

import threading
import time
import logging
import numpy as np
from typing import Callable, Optional, Any

logger = logging.getLogger(__name__)

class SharedStreamListener:
    """
    Simple mock implementation of a stream listener for testing.
    """
    
    def __init__(self, 
                source_path: str,
                push_audio: Optional[Callable] = None,
                buffer_size: int = 4096,
                sample_rate: int = 16000):
        """
        Initialize the stream listener.
        
        Args:
            source_path: Path to the audio/video source
            push_audio: Callback for audio data
            buffer_size: Audio buffer size
            sample_rate: Audio sample rate
        """
        self.source_path = source_path
        self.push_audio = push_audio
        self.buffer_size = buffer_size
        self.sample_rate = sample_rate
        
        self.running = False
        self.thread = None
        
        # Log initialization
        logger.info(f"Initialized SharedStreamListener with source: {source_path}")
    
    def start(self):
        """Start streaming audio from the source."""
        if self.thread and self.running:
            logger.warning("Listener already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._stream_simulation)
        self.thread.daemon = True
        self.thread.start()
        
        logger.info("Started stream listener")
    
    def stop(self):
        """Stop streaming audio."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        
        logger.info("Stopped stream listener")
    
    def _stream_simulation(self):
        """Simulate streaming audio data."""
        if not self.push_audio:
            logger.warning("No audio push callback set")
            return
        
        try:
            # Simulate reading from the source file
            logger.info(f"Started streaming from {self.source_path}")
            
            # Generate fake audio buffers
            duration = 60  # Simulate 60 seconds of audio
            start_time = time.time()
            
            for i in range(int(duration)):
                if not self.running:
                    break
                
                # Generate empty audio buffer of the right size
                # In a real implementation this would be actual audio data
                buffer_duration = 1.0  # 1 second of audio
                num_samples = int(self.sample_rate * buffer_duration)
                audio_buffer = np.zeros(num_samples, dtype=np.int16)
                
                # Push the audio buffer to the callback
                current_time = time.time() - start_time
                if self.push_audio:
                    self.push_audio(audio_buffer, timestamp=current_time)
                
                # Slow down to real-time
                time.sleep(0.5)  # Speed up simulation slightly
            
            logger.info("Finished streaming audio")
            
        except Exception as e:
            logger.error(f"Error in stream simulation: {str(e)}")
    
    def restart(self):
        """Restart the stream listener."""
        self.stop()
        time.sleep(0.5)  # Short delay to ensure cleanup
        self.start()
