import ffmpeg
import numpy as np
import time
import logging
import threading
import queue
from typing import Callable, Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)

class SharedStreamListener:
    """
    Read local video file -> output audio chunks to push_audio callback with advanced buffering.
    
    Features:
    - Accurate audio timestamps by tracking playback position
    - Buffer management for continuous streaming
    - Separate processing thread to prevent blocking
    - Error handling and recovery
    - Stream status reporting
    """
    def __init__(
        self, 
        filepath: str, 
        push_audio: Callable, 
        sample_rate: int = 16000, 
        chunk_sec: float = 0.5,
        buffer_size: int = 20,  # Buffer 10 seconds by default (20 * 0.5s chunks)
        retry_on_error: bool = True,
        max_retries: int = 3
    ):
        self.file = filepath
        self.push_audio = push_audio
        self.sample_rate = sample_rate
        self.chunk_sec = chunk_sec
        self.buffer_size = buffer_size
        self.retry_on_error = retry_on_error
        self.max_retries = max_retries
        
        # Playback state
        self.start_time = None
        self.current_position = 0.0  # Position in seconds
        self.duration = 0.0  # Duration in seconds
        self.format_info = None
        
        # Buffering system
        self.audio_buffer = queue.Queue(maxsize=buffer_size)
        self.processing_thread = None
        self.reader_thread = None
        self.stop_flag = threading.Event()
        self.error = None
        
        # Performance metrics
        self.buffer_level = 0
        self.underruns = 0
        self.overruns = 0
        self.last_stats_time = 0

    def start(self):
        """Start processing the stream in separate threads for improved reliability."""
        try:
            # Get video duration and actual format info
            probe = ffmpeg.probe(self.file)
            self.format_info = probe['format']
            self.duration = float(self.format_info.get('duration', 0))
            logger.info(f"Starting stream: {self.file} (duration: {self.duration:.2f}s)")
            
            # Reset state
            self.current_position = 0.0
            self.stop_flag.clear()
            self.error = None
            self.underruns = 0
            self.overruns = 0
            
            # Start reader thread
            self.reader_thread = threading.Thread(
                target=self._read_stream,
                daemon=True
            )
            self.reader_thread.start()
            
            # Start processor thread
            self.processing_thread = threading.Thread(
                target=self._process_buffer,
                daemon=True
            )
            self.processing_thread.start()
            
            # Wait for both threads to complete
            self.processing_thread.join()
            self.reader_thread.join()
            
            if self.error:
                raise RuntimeError(f"Stream processing error: {self.error}")
            
            logger.info(f"Stream processing complete. Underruns: {self.underruns}, Overruns: {self.overruns}")
        except Exception as e:
            logger.error(f"Error processing audio stream: {str(e)}")
            self.error = str(e)
            self.stop_flag.set()
            raise
            
    def stop(self):
        """Stop the stream processing."""
        logger.info("Stopping stream processing")
        self.stop_flag.set()
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
            
        if self.reader_thread and self.reader_thread.is_alive():
            self.reader_thread.join(timeout=2.0)
            
    def get_status(self) -> Dict[str, Any]:
        """Get current stream status information."""
        return {
            "current_position": self.current_position,
            "duration": self.duration,
            "progress_pct": (self.current_position / self.duration * 100) if self.duration else 0,
            "buffer_level": self.buffer_level,
            "buffer_capacity": self.buffer_size,
            "buffer_pct": (self.buffer_level / self.buffer_size * 100) if self.buffer_size else 0,
            "underruns": self.underruns,
            "overruns": self.overruns,
            "error": self.error
        }
            
    def _read_stream(self):
        """Reader thread: reads from ffmpeg and fills the buffer."""
        retry_count = 0
        while not self.stop_flag.is_set() and retry_count <= self.max_retries:
            process = None
            try:
                # Start the ffmpeg process
                process = (
                    ffmpeg
                    .input(self.file)
                    .output('pipe:', format='s16le', ac=1, ar=str(self.sample_rate))
                    .run_async(pipe_stdout=True, quiet=True)
                )
                
                bytes_per_chunk = int(self.sample_rate * self.chunk_sec * 2)  # int16 -> 2 bytes
                position = 0.0  # Local position tracking
                
                while not self.stop_flag.is_set():
                    data = process.stdout.read(bytes_per_chunk)
                    if not data:
                        # End of stream
                        logger.info("End of stream reached")
                        self.stop_flag.set()
                        break
                        
                    # Convert to numpy array
                    audio_chunk = np.frombuffer(data, np.int16)
                    
                    # Try to put in buffer, with timeout to check stop_flag periodically
                    try:
                        self.audio_buffer.put((audio_chunk, position), timeout=1.0)
                        position += self.chunk_sec
                    except queue.Full:
                        # Buffer is full, log overrun
                        self.overruns += 1
                        if self.overruns % 10 == 1:  # Log every 10th overrun to avoid spam
                            logger.warning(f"Buffer overrun ({self.overruns} total)")
                
                # Process completed normally
                retry_count = self.max_retries + 1  # Exit the retry loop
            except Exception as e:
                logger.error(f"Error in reader thread: {str(e)}")
                retry_count += 1
                
                if retry_count <= self.max_retries and self.retry_on_error:
                    logger.info(f"Retrying stream read ({retry_count}/{self.max_retries})")
                    time.sleep(1.0)  # Brief delay before retry
                else:
                    self.error = str(e)
                    self.stop_flag.set()
            finally:
                if process is not None:
                    process.kill()
                    
    def _process_buffer(self):
        """Processor thread: takes from buffer and sends to callback."""
        self.start_time = time.time()
        self.last_stats_time = self.start_time
        
        while not self.stop_flag.is_set():
            try:
                # Update buffer level for monitoring
                self.buffer_level = self.audio_buffer.qsize()
                
                # Log buffer stats periodically
                current_time = time.time()
                if current_time - self.last_stats_time > 10.0:  # Every 10 seconds
                    logger.debug(f"Buffer status: {self.buffer_level}/{self.buffer_size} " +
                                f"chunks, {self.underruns} underruns, {self.overruns} overruns")
                    self.last_stats_time = current_time
                
                # Get chunk from buffer with timeout to check stop_flag periodically
                try:
                    audio_chunk, timestamp = self.audio_buffer.get(timeout=1.0)
                    self.current_position = timestamp
                    
                    # Push audio with timestamp
                    self.push_audio(audio_chunk, timestamp)
                    
                except queue.Empty:
                    if not self.stop_flag.is_set():
                        # Buffer underrun - only count as error if we're not at the end
                        self.underruns += 1
                        if self.underruns % 10 == 1:  # Log every 10th underrun
                            logger.warning(f"Buffer underrun ({self.underruns} total)")
            except Exception as e:
                logger.error(f"Error in processor thread: {str(e)}")
                self.error = str(e)
                self.stop_flag.set()
