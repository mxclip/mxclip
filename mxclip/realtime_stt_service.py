"""
Simplified Speech-to-Text service for testing.
"""

import logging
import time
import threading
import numpy as np
from typing import Callable, Optional, Dict, Any, List

logger = logging.getLogger(__name__)

class AudioToTextRecorder:
    """
    Mock implementation of AudioToTextRecorder for testing.
    
    This mocks the behavior of a real speech recognition system.
    """
    
    def __init__(
        self,
        model: str = "base.en",
        language: str = "en",
        use_microphone: bool = False,
        device: str = "cpu",
        enable_realtime_transcription: bool = True,
        on_realtime_transcription_update: Optional[Callable[[str], None]] = None,
        auto_reconnect: bool = True,
        **kwargs
    ):
        """
        Initialize the audio to text recorder.
        
        Args:
            model: Model name to use for transcription
            language: Language code
            use_microphone: Whether to use microphone input
            device: Device to run inference on
            enable_realtime_transcription: Whether to enable real-time transcription
            on_realtime_transcription_update: Callback for transcription updates
            auto_reconnect: Whether to automatically reconnect on error
            **kwargs: Additional arguments
        """
        self.model = model
        self.language = language
        self.use_microphone = use_microphone
        self.device = device
        self.enable_realtime_transcription = enable_realtime_transcription
        self.on_realtime_transcription_update = on_realtime_transcription_update
        self.auto_reconnect = auto_reconnect
        
        self.running = True
        self.audio_buffer = []
        self.buffer_timer = None
        self.last_flush_time = time.time()
        
        # Sample transcriptions to simulate responses
        self.sample_texts = [
            "Hello, this is a test.", 
            "Wow, that was amazing!",
            "I can't believe how cool that was.",
            "That is so awesome!",
            "Nice job, that was incredible."
        ]
        
        logger.info(f"Initialized AudioToTextRecorder with model {model}")
        
        # Start buffer processing
        if self.enable_realtime_transcription:
            self._start_buffer_timer()
    
    def feed_audio(self, audio_chunk: np.ndarray) -> None:
        """
        Feed audio data to the recorder.
        
        Args:
            audio_chunk: Audio data (numpy array)
        """
        self.audio_buffer.append(audio_chunk)
        
        # If we have enough audio, process it
        if len(self.audio_buffer) >= 5:  # About 1 second of audio
            self._process_buffer()
    
    def _start_buffer_timer(self) -> None:
        """Start the timer to periodically process the buffer."""
        if not self.running:
            return
            
        # Process the buffer
        self._process_buffer()
        
        # Schedule the next processing
        self.buffer_timer = threading.Timer(1.0, self._start_buffer_timer)
        self.buffer_timer.daemon = True
        self.buffer_timer.start()
    
    def _process_buffer(self) -> None:
        """Process the audio buffer and generate transcriptions."""
        if not self.audio_buffer or not self.running:
            return
            
        # In a real implementation, we would process the audio here
        # For this mock version, we randomly choose a sample text
        if self.on_realtime_transcription_update and np.random.random() < 0.3:
            text = np.random.choice(self.sample_texts)
            self.on_realtime_transcription_update(text)
            
        # Clear the buffer
        self.audio_buffer = []
        self.last_flush_time = time.time()
    
    def shutdown(self) -> None:
        """Shut down the recorder and release resources."""
        self.running = False
        
        if self.buffer_timer:
            self.buffer_timer.cancel()
            
        logger.info("Shutting down AudioToTextRecorder")


class RTSTTService:
    """
    Simplified Speech-to-Text Service for testing.
    """
    
    def __init__(self, callback: Callable[[str, Optional[float]], None], model_size: str = "base.en"):
        """
        Initialize the service.
        
        Args:
            callback: Function to call with transcription results
            model_size: Size of the model to use for transcription
        """
        self.text_cb = callback
        self.current_timestamp = None
        
        # Create a wrapper callback to include timestamp
        def transcription_callback(text):
            self.text_cb(text, self.current_timestamp)
        
        # Initialize the audio to text recorder
        self.recorder = AudioToTextRecorder(
            model=model_size,
            language="en",
            use_microphone=False,
            enable_realtime_transcription=True,
            on_realtime_transcription_update=transcription_callback
        )
        
        logger.info(f"Initialized RTSTTService with model size {model_size}")
    
    def push(self, audio: np.ndarray, timestamp: Optional[float] = None) -> None:
        """
        Push audio data to the service.
        
        Args:
            audio: Audio data (numpy array)
            timestamp: Timestamp of the audio (optional)
        """
        if timestamp is not None:
            self.current_timestamp = timestamp
            
        self.recorder.feed_audio(audio)
    
    def shutdown(self) -> None:
        """
        Shut down the service and release resources.
        """
        self.recorder.shutdown()
        logger.info("Shutting down RTSTTService")
