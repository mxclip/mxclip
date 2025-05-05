# MX Clipping 1.0

Real-time AI-powered clipping and highlight detection for video content.

## Overview

MX Clipping is a Python-based system for real-time audio/video processing and automatic clip generation based on various triggers such as:

- Keyword detection in speech
- Repeated phrases
- Chat activity spikes
- Audio content analysis (NEW)
- Emotional moment detection (NEW)
- Platform stream recording (NEW)

## Features

- **Real-Time Speech-to-Text**: Convert audio to text in real-time using Whisper models via RealtimeSTT
- **Multi-User Configuration**: Support for different users with their own keyword preferences
- **Trigger Detection**: Identify clip-worthy moments based on speech, repetition, or chat activity
- **Automatic Clip Generation**: Extract, process, and save video clips with subtitles
- **Metadata Management**: Store organized metadata for each clip
- **Advanced Audio Analysis**: Analyze audio content for emotional tone, keywords, and key moments (NEW)
- **Intelligent Clip Suggestion**: Get smart clip suggestions based on audio content understanding (NEW)
- **Platform Stream Recording**: Record streams directly from Twitch, YouTube, and more with URL resolution (NEW)

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
- `KimiAudioProcessor`: Advanced audio processing and understanding with Kimi-Audio (NEW)
- `ClipSuggester`: Smart clip suggestions based on content relevance (NEW)
- `StreamResolver`: Resolution of platform URLs to direct stream URLs (NEW)
- `LiveRecordingService`: Recording from streaming platforms (NEW)

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

### Advanced Audio Analysis (NEW)

```python
from mxclip.audio_processor import KimiAudioProcessor

# Initialize the Kimi-Audio processor
processor = KimiAudioProcessor()

# Transcribe audio with timestamps
transcription = processor.transcribe_audio("video.mp4")
print(f"Transcription: {transcription['text']}")

# Analyze audio content
analysis = processor.analyze_audio_content("video.mp4")
print(f"Emotion: {analysis.get('emotion')}")
print(f"Tone: {analysis.get('tone')}")

# Generate an audio caption
caption = processor.generate_audio_caption("video.mp4")
print(f"Caption: {caption}")
```

### Intelligent Clip Suggestion (NEW)

```python
from mxclip.clip_suggestion import ClipSuggester

# Initialize the clip suggester
suggester = ClipSuggester()

# Get clip suggestions
suggestions = suggester.suggest_clips(
    media_path="video.mp4",
    keywords=["exciting", "amazing", "victory"],
    max_suggestions=5
)

# Process suggestions
for i, suggestion in enumerate(suggestions):
    print(f"Suggestion #{i+1}:")
    print(f"  Time: {suggestion['start']} - {suggestion['end']}")
    print(f"  Reason: {suggestion['reason']}")
    print(f"  Score: {suggestion['score']}")
```

### Stream Recording (NEW)

```python
from mxclip.stream_resolver import StreamResolver
from mxclip.live_recording_service import LiveRecordingService

# Resolve a platform URL
resolver = StreamResolver()
direct_url, error = resolver.resolve_stream_url("https://twitch.tv/username")

# Record from a platform URL
def on_recording_complete(output_file, success, error_message):
    print(f"Recording completed: {output_file}")

recorder = LiveRecordingService(
    user_url="https://twitch.tv/username",
    output_dir="recordings",
    segment_time=300,  # 5 minute segments
    completion_callback=on_recording_complete
)

# Start recording
recorder.start()

# Later, stop recording
recorder.stop()
```

## Command Line Interface (CLI)

MX Clipping includes a command-line interface for common tasks:

```bash
# Record microphone for 5 seconds and transcribe
python -m mxclip.main record --duration 5

# Play a local video file and transcribe its audio in real time
python -m mxclip.main video --file sample.mp4

# Run video analysis with chat trigger and clip generation
python -m mxclip.main analyze --video sample.mp4 --chat-freq 0.2

# Use Kimi-Audio to suggest optimal clip points (NEW)
python -m mxclip.main suggest --video sample.mp4 --keywords "highlight,amazing"

# Record from a streaming platform URL (NEW)
python -m mxclip.main stream --url https://twitch.tv/username
```

## Integration Example

See the `examples/integration_example.py` file for a complete example of how to integrate all components together.

## License

[MIT License](LICENSE)
