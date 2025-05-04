"""
Trigger Detector module for detecting various clip-worthy moments.

This module provides detection for:
1. Keyword matching in transcriptions
2. Repeated phrases detection
3. Chat activity spikes
"""

import re
import time
import logging
from typing import List, Dict, Any, Callable, Optional, Set
from collections import deque

logger = logging.getLogger(__name__)

class TriggerEvent:
    """
    Represents a trigger event that may lead to clip creation.
    """
    def __init__(self, timestamp: float, reason: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a trigger event.
        
        Args:
            timestamp: The timestamp when the trigger occurred
            reason: The reason for the trigger (e.g., keyword match, repetition)
            metadata: Additional metadata related to the trigger
        """
        self.timestamp = timestamp
        self.reason = reason
        self.metadata = metadata or {}


class TriggerDetector:
    """
    Detector for clip-worthy moments based on various criteria.
    
    Features:
    - Keyword matching in transcribed text
    - Detection of repeated phrases
    - Chat activity spike detection
    """
    
    def __init__(
        self,
        callback: Optional[Callable[[TriggerEvent], None]] = None,
        keywords: Optional[List[str]] = None,
        enable_repeat_check: bool = True,
        repeat_window_seconds: float = 10.0,
        repeat_threshold: int = 2,
        enable_chat_check: bool = True,
        chat_activity_threshold: int = 15
    ):
        """
        Initialize the trigger detector.
        
        Args:
            callback: Function to call when a trigger is detected
            keywords: List of keywords to match in transcriptions
            enable_repeat_check: Whether to enable repeated phrase detection
            repeat_window_seconds: Time window to check for repetitions
            repeat_threshold: Number of repetitions needed to trigger
            enable_chat_check: Whether to enable chat activity detection
            chat_activity_threshold: Number of messages needed in the time window to trigger
        """
        self.callback = callback
        self.keywords = keywords or []
        
        # Compiled regex patterns for faster matching
        self.keyword_patterns = [re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE) 
                                for kw in self.keywords]
        
        # Repetition detection
        self.enable_repeat_check = enable_repeat_check
        self.repeat_window_seconds = repeat_window_seconds
        self.repeat_threshold = repeat_threshold
        self.recent_phrases = deque()
        
        # Chat activity detection
        self.enable_chat_check = enable_chat_check
        self.chat_activity_threshold = chat_activity_threshold
        self.chat_messages = deque()
        
        logger.info(f"Initialized TriggerDetector with {len(self.keywords)} keywords")
    
    def set_callback(self, callback: Callable[[TriggerEvent], None]) -> None:
        """
        Set the callback function for trigger events.
        
        Args:
            callback: Function to call when a trigger is detected
        """
        self.callback = callback
    
    def update_keywords(self, keywords: List[str]) -> None:
        """
        Update the list of keywords to detect.
        
        Args:
            keywords: New list of keywords
        """
        self.keywords = keywords
        self.keyword_patterns = [re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE) 
                                for kw in self.keywords]
        logger.info(f"Updated keywords: now watching {len(self.keywords)} terms")
    
    def process_transcription(self, timestamp: float, text: str) -> Optional[TriggerEvent]:
        """
        Process a transcription to check for triggers.
        
        Args:
            timestamp: Timestamp of the transcription
            text: Transcribed text
        
        Returns:
            TriggerEvent if a trigger was detected, None otherwise
        """
        # Check for keyword matches
        for i, pattern in enumerate(self.keyword_patterns):
            if pattern.search(text):
                logger.info(f"Keyword match: '{self.keywords[i]}' at {timestamp:.2f}s")
                trigger = TriggerEvent(
                    timestamp=timestamp,
                    reason=f"keyword:{self.keywords[i]}",
                    metadata={
                        "text": text,
                        "keyword": self.keywords[i]
                    }
                )
                if self.callback:
                    self.callback(trigger)
                return trigger
        
        # Check for repetitions if enabled
        if self.enable_repeat_check and text:
            trigger = self._check_repetition(timestamp, text)
            if trigger and self.callback:
                self.callback(trigger)
                return trigger
        
        return None
    
    def _check_repetition(self, timestamp: float, text: str) -> Optional[TriggerEvent]:
        """
        Check if a phrase is repeated within the time window.
        
        Args:
            timestamp: Timestamp of the current phrase
            text: The phrase to check
        
        Returns:
            TriggerEvent if a repetition was detected, None otherwise
        """
        # Clean the text for comparison (lowercase, remove extra spaces)
        clean_text = ' '.join(text.lower().split())
        if not clean_text:
            return None
        
        # Add the current phrase to the list
        self.recent_phrases.append((timestamp, clean_text))
        
        # Remove phrases outside the time window
        while (self.recent_phrases and 
               self.recent_phrases[0][0] < timestamp - self.repeat_window_seconds):
            self.recent_phrases.popleft()
        
        # Count occurrences of the current phrase in the window
        count = sum(1 for _, phrase in self.recent_phrases if phrase == clean_text)
        
        if count >= self.repeat_threshold:
            logger.info(f"Repetition detected: '{clean_text}' repeated {count} times within "
                        f"{self.repeat_window_seconds}s window")
            return TriggerEvent(
                timestamp=timestamp,
                reason=f"repetition:{clean_text}",
                metadata={
                    "text": clean_text,
                    "count": count,
                    "window": self.repeat_window_seconds
                }
            )
        
        return None
    
    def process_chat(self, timestamp: float, message: str, username: str) -> Optional[TriggerEvent]:
        """
        Process a chat message to check for chat activity spikes.
        
        Args:
            timestamp: Timestamp of the message
            message: The chat message content
            username: The username of the sender
        
        Returns:
            TriggerEvent if a chat activity spike was detected, None otherwise
        """
        if not self.enable_chat_check:
            return None
        
        # Add the current message to the list
        self.chat_messages.append((timestamp, message, username))
        
        # Remove messages outside the time window
        while (self.chat_messages and 
               self.chat_messages[0][0] < timestamp - self.repeat_window_seconds):
            self.chat_messages.popleft()
        
        # Count messages in the window
        message_count = len(self.chat_messages)
        
        if message_count >= self.chat_activity_threshold:
            logger.info(f"Chat activity spike: {message_count} messages within "
                        f"{self.repeat_window_seconds}s window")
            
            # Get sample of recent messages for context
            recent_msgs = [(msg_ts, msg, user) for msg_ts, msg, user in 
                          list(self.chat_messages)[-5:]]
            
            trigger = TriggerEvent(
                timestamp=timestamp,
                reason="chat_activity",
                metadata={
                    "message_count": message_count,
                    "window": self.repeat_window_seconds,
                    "recent_messages": recent_msgs
                }
            )
            
            if self.callback:
                self.callback(trigger)
            return trigger
        
        return None 