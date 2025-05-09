"""
Create a test video file for the integration example.
"""
import numpy as np
import subprocess
import os
from PIL import Image

def create_test_video():
    # Create directories
    os.makedirs('test_data', exist_ok=True)
    
    # Create images for video frames
    frames_dir = os.path.join('test_data', 'frames')
    os.makedirs(frames_dir, exist_ok=True)
    
    # Make 30 simple frames (1 second at 30fps)
    print("Creating frames...")
    for i in range(30):
        # Create a colored image that changes slightly each frame
        img = Image.new('RGB', (640, 480), color=(255, i*8 % 256, 100))
        frame_path = os.path.join(frames_dir, f'frame_{i:03d}.png')
        img.save(frame_path)
    
    # Create video from frames using ffmpeg
    print("Creating video...")
    video_path = os.path.join('test_data', 'test_video.mp4')
    subprocess.run([
        'ffmpeg', '-y',
        '-framerate', '30',
        '-i', os.path.join(frames_dir, 'frame_%03d.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-t', '1',  # 1 second duration
        video_path
    ], check=True)
    
    # Add silent audio track
    print("Adding audio track...")
    final_video_path = os.path.join('test_data', 'test_video_with_audio.mp4')
    subprocess.run([
        'ffmpeg', '-y',
        '-f', 'lavfi',
        '-i', 'anullsrc=r=44100:cl=mono',
        '-i', video_path,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-shortest',
        final_video_path
    ], check=True)
    
    print(f"Created test video: {final_video_path}")
    return final_video_path

if __name__ == "__main__":
    create_test_video() 