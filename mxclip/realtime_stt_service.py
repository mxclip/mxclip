from RealtimeSTT import AudioToTextRecorder
from typing import Callable, Optional, Any, List

class RTSTTService:
    """
    Asynchronous real-time speech-to-text service:
      push(samples, timestamp) → internal thread calls RealtimeSTT
      samples should be 16kHz mono PCM int16 numpy array
      
    Preserves audio timestamps for accurate transcription timing.
    """
    def __init__(self, text_callback, model_size="base.en"):
        self.text_cb = text_callback
        self.current_timestamp = None
        
        # Wrap the callback to include timestamp
        def callback_with_timestamp(text):
            self.text_cb(text, self.current_timestamp)
            
        self.recorder = AudioToTextRecorder(
            model=model_size,
            language="en",
            use_microphone=False,
            spinner=False,
            enable_realtime_transcription=True,
            on_realtime_transcription_update=callback_with_timestamp
        )

    def push(self, samples, timestamp: Optional[float] = None):
        """
        Push audio samples with optional timestamp to the STT service.
        
        Args:
            samples: Audio samples (16kHz mono PCM int16 numpy array)
            timestamp: Timestamp of the audio in seconds from start of stream
        """
        # Store timestamp for use in callback
        self.current_timestamp = timestamp
        self.recorder.feed_audio(samples)

    def shutdown(self):
        self.recorder.shutdown()
