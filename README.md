# MX Clipping 0.1

Real-time AI-powered clipping and highlight detection for video content.

## Overview

MX Clipping is a Python-based system for real-time audio/video processing and automatic clip generation based on various triggers such as:

- Keyword detection in speech
- Repeated phrases
- Chat activity spikes
- And more!

## Features

- **Real-Time Speech-to-Text**: Convert audio to text in real-time using Whisper models via RealtimeSTT
- **Multi-User Configuration**: Support for different users with their own keyword preferences
- **Trigger Detection**: Identify clip-worthy moments based on speech, repetition, or chat activity
- **Automatic Clip Generation**: Extract, process, and save video clips with subtitles
- **Metadata Management**: Store organized metadata for each clip

## Components

### Core Components

- `RTSTTService`: Real-time speech-to-text service
- `SharedStreamListener`: Audio extraction from video streams
- `ClipService`: Video clip generation with subtitles
- `UserProcessor`: User-specific trigger processing

### New Components

- `TriggerDetector`: Detection of keywords, repeated phrases, and chat activity spikes
- `ConfigLoader`: Management of user-specific configurations
- `ClipsOrganizer`: Organization of clips with consistent naming and metadata

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mx-clipping.git
   cd mx-clipping
   ```

2. Install dependencies:
   ```bash
   pip install -e .
   ```

## Usage

### Basic Usage

```python
from mxclip.realtime_stt_service import RTSTTService
from mxclip.shared_stream_listener import SharedStreamListener

# Initialize the STT service
def on_text(text):
    print(f"Transcribed: {text}")

stt = RTSTTService(on_text)

# Create a stream listener and attach it to the STT service
listener = SharedStreamListener("sample.mp4", push_audio=stt.push)
listener.start()

# Don't forget to clean up
stt.shutdown()
```

### Trigger Detection Example

```python
from mxclip.trigger_detector import TriggerDetector

# Create a detector with custom keywords
detector = TriggerDetector(
    keywords=["awesome", "amazing", "wow"],
    enable_repeat_check=True,
    repeat_threshold=2
)

# Set up a callback for triggers
def on_trigger(event):
    print(f"Trigger detected: {event.reason} at {event.timestamp}")

detector.set_callback(on_trigger)

# Process a transcription
detector.process_transcription(timestamp=123.45, text="That was really awesome!")
```

### User Configuration Example

```python
from mxclip.config_loader import ConfigLoader

# Create a config loader
config_loader = ConfigLoader(config_dir="configs")

# Load or create a user's config
user_config = config_loader.load_config("user1")

# Update a user's config
config_loader.update_config("user1", {
    "keywords": ["epic", "awesome", "incredible"],
    "enable_chat_check": True
})
```

### Clips Organization Example

```python
from mxclip.clips_organizer import ClipsOrganizer

# Create a clips organizer
organizer = ClipsOrganizer(base_dir="clips")

# Generate paths for a new clip
video_path, metadata_path = organizer.get_clip_path(
    user_id="user1",
    streamer_id="streamer123"
)

# Generate and save metadata
metadata = organizer.generate_metadata(
    user_id="user1",
    streamer_id="streamer123",
    trigger_time=123.45,
    clip_start=108.45,
    clip_end=138.45,
    trigger_reason="keyword:awesome"
)

organizer.save_clip_metadata(metadata_path, metadata)
```

## Integration Example

See the `examples/integration_example.py` file for a complete example of how to integrate all components together.

## License

[MIT License](LICENSE)
