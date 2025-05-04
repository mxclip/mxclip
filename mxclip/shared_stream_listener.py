import ffmpeg
import numpy as np
import time

class SharedStreamListener:
    """
    Read local video file -> output 0.5s audio chunks to push_audio callback
    
    Provides accurate audio timestamps by tracking playback position.
    """
    def __init__(self, filepath, push_audio, sample_rate=16000, chunk_sec=0.5):
        self.file = filepath
        self.push_audio = push_audio
        self.sample_rate = sample_rate
        self.chunk_sec = chunk_sec
        self.start_time = None
        self.current_position = 0.0  # Position in seconds

    def start(self):
        try:
            # Get video duration and actual format info
            probe = ffmpeg.probe(self.file)
            self.format_info = probe['format']
            self.duration = float(self.format_info.get('duration', 0))
            
            process = (
                ffmpeg
                .input(self.file)
                .output('pipe:', format='s16le', ac=1, ar=str(self.sample_rate))
                .run_async(pipe_stdout=True, quiet=True)
            )
            
            bytes_per_chunk = int(self.sample_rate * self.chunk_sec * 2)  # int16 -> 2 bytes
            self.start_time = time.time()
            
            while True:
                data = process.stdout.read(bytes_per_chunk)
                if not data:
                    break
                    
                # Calculate actual audio timestamp
                audio_chunk = np.frombuffer(data, np.int16)
                audio_timestamp = self.current_position
                
                # Push audio with timestamp
                self.push_audio(audio_chunk, audio_timestamp)
                
                # Update position for next chunk
                self.current_position += self.chunk_sec
        except Exception as e:
            print(f"Error processing audio stream: {str(e)}")
            raise
        finally:
            if 'process' in locals():
                process.kill()
