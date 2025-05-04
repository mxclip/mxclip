"""
Tests for the TriggerDetector module.
"""

import time
import pytest
from mxclip.trigger_detector import TriggerDetector, TriggerEvent

def test_keyword_detection():
    """Test keyword matching in transcriptions."""
    detected_triggers = []
    
    def callback(trigger):
        detected_triggers.append(trigger)
    
    # Initialize detector with some keywords
    detector = TriggerDetector(
        callback=callback,
        keywords=["hello", "awesome", "let's go"],
        enable_repeat_check=False,
        enable_chat_check=False
    )
    
    # Test with non-matching text
    timestamp = time.time()
    result = detector.process_transcription(timestamp, "This is a test")
    assert result is None
    assert len(detected_triggers) == 0
    
    # Test with matching text
    timestamp = time.time()
    result = detector.process_transcription(timestamp, "Hello world")
    assert result is not None
    assert result.reason == "keyword:hello"
    assert result.metadata["keyword"] == "hello"
    assert result.metadata["text"] == "Hello world"
    assert len(detected_triggers) == 1
    
    # Test with another keyword
    timestamp = time.time()
    result = detector.process_transcription(timestamp, "That was awesome")
    assert result is not None
    assert result.reason == "keyword:awesome"
    assert result.metadata["keyword"] == "awesome"
    assert len(detected_triggers) == 2
    
    # Test with phrase keyword
    timestamp = time.time()
    result = detector.process_transcription(timestamp, "Let's go to the next level")
    assert result is not None
    assert result.reason == "keyword:let's go"
    assert result.metadata["keyword"] == "let's go"
    assert len(detected_triggers) == 3

def test_repetition_detection():
    """Test repeated phrase detection."""
    detected_triggers = []
    
    def callback(trigger):
        detected_triggers.append(trigger)
    
    # Initialize detector with repetition detection enabled
    detector = TriggerDetector(
        callback=callback,
        keywords=[],
        enable_repeat_check=True,
        repeat_window_seconds=5.0,
        repeat_threshold=2,
        enable_chat_check=False
    )
    
    # First occurrence of a phrase
    timestamp = time.time()
    result = detector.process_transcription(timestamp, "This is a test phrase")
    assert result is None
    assert len(detected_triggers) == 0
    
    # Second occurrence of the same phrase within window - should trigger
    timestamp = time.time()
    result = detector.process_transcription(timestamp, "This is a test phrase")
    assert result is not None
    assert result.reason.startswith("repetition:")
    assert result.metadata["count"] == 2
    assert len(detected_triggers) == 1
    
    # Different phrase - should not trigger
    timestamp = time.time()
    result = detector.process_transcription(timestamp, "This is a different phrase")
    assert result is None
    assert len(detected_triggers) == 1
    
    # Test with phrases outside time window
    detector = TriggerDetector(
        callback=callback,
        keywords=[],
        enable_repeat_check=True,
        repeat_window_seconds=1.0,  # 1 second window
        repeat_threshold=2,
        enable_chat_check=False
    )
    
    # First occurrence
    timestamp = time.time()
    detector.process_transcription(timestamp, "Time window test")
    
    # Wait longer than the window
    time.sleep(1.5)
    
    # Second occurrence - should not trigger as outside window
    timestamp = time.time()
    result = detector.process_transcription(timestamp, "Time window test")
    assert result is None
    
    # Rapid repetition - should trigger
    timestamp = time.time()
    detector.process_transcription(timestamp, "Rapid test")
    timestamp = time.time()
    result = detector.process_transcription(timestamp, "Rapid test")
    assert result is not None
    assert result.reason.startswith("repetition:")

def test_chat_activity_detection():
    """Test chat activity spike detection."""
    detected_triggers = []
    
    def callback(trigger):
        detected_triggers.append(trigger)
    
    # Initialize detector with chat activity detection enabled
    detector = TriggerDetector(
        callback=callback,
        keywords=[],
        enable_repeat_check=False,
        enable_chat_check=True,
        chat_activity_threshold=3  # Low threshold for testing
    )
    
    # Send a few chat messages
    timestamp = time.time()
    detector.process_chat(timestamp, "Hello", "user1")
    assert len(detected_triggers) == 0
    
    timestamp = time.time()
    detector.process_chat(timestamp, "Hi there", "user2")
    assert len(detected_triggers) == 0
    
    # Third message should trigger
    timestamp = time.time()
    result = detector.process_chat(timestamp, "What's up?", "user3")
    assert result is not None
    assert result.reason == "chat_activity"
    assert result.metadata["message_count"] == 3
    assert len(detected_triggers) == 1
    
    # Test metadata contains recent messages
    recent_messages = result.metadata.get("recent_messages", [])
    assert len(recent_messages) == 3
    assert recent_messages[2][1] == "What's up?"
    assert recent_messages[2][2] == "user3"

def test_update_keywords():
    """Test updating keywords dynamically."""
    detector = TriggerDetector(keywords=["initial"])
    
    # Test initial keyword
    timestamp = time.time()
    result = detector.process_transcription(timestamp, "This is the initial test")
    assert result is not None
    assert result.reason == "keyword:initial"
    
    # Update keywords
    detector.update_keywords(["updated", "new"])
    
    # Initial keyword should no longer match
    timestamp = time.time()
    result = detector.process_transcription(timestamp, "This is the initial test")
    assert result is None
    
    # New keywords should match
    timestamp = time.time()
    result = detector.process_transcription(timestamp, "This is an updated test")
    assert result is not None
    assert result.reason == "keyword:updated"
    
    timestamp = time.time()
    result = detector.process_transcription(timestamp, "This is a new test")
    assert result is not None
    assert result.reason == "keyword:new" 