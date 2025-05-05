"""
Clip suggestion module using Kimi-Audio for MX Clipping.

This module uses Kimi-Audio to analyze audio content and suggest optimal
clip boundaries based on content understanding and relevance.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any
import json
import tempfile
import subprocess

from .audio_processor import KimiAudioProcessor

logger = logging.getLogger(__name__)

class ClipSuggester:
    """
    Suggests optimal clip boundaries using Kimi-Audio's understanding of content.
    
    Features:
    - Content relevance scoring
    - Key moment detection
    - Highlight detection based on audio cues
    - Emotional moment identification
    """
    
    def __init__(
        self,
        audio_processor: Optional[KimiAudioProcessor] = None,
        min_clip_duration: float = 5.0,
        max_clip_duration: float = 60.0,
        relevance_threshold: float = 0.5
    ):
        """
        Initialize the clip suggester.
        
        Args:
            audio_processor: KimiAudioProcessor instance to use for audio analysis
            min_clip_duration: Minimum clip duration in seconds
            max_clip_duration: Maximum clip duration in seconds
            relevance_threshold: Threshold for content relevance (0.0-1.0)
        """
        # Create audio processor if not provided
        if audio_processor is None:
            try:
                self.audio_processor = KimiAudioProcessor()
            except Exception as e:
                logger.error(f"Failed to initialize audio processor: {str(e)}")
                raise
        else:
            self.audio_processor = audio_processor
            
        self.min_clip_duration = min_clip_duration
        self.max_clip_duration = max_clip_duration
        self.relevance_threshold = relevance_threshold
    
    def suggest_clips(
        self,
        media_path: str,
        keywords: Optional[List[str]] = None,
        max_suggestions: int = 5,
        extract_audio: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Analyze media and suggest optimal clip boundaries.
        
        Args:
            media_path: Path to the media file (video or audio)
            keywords: Optional list of keywords to prioritize
            max_suggestions: Maximum number of clip suggestions to return
            extract_audio: Whether to extract audio from video files
            
        Returns:
            List of clip suggestion dictionaries with start/end times and metadata
        """
        try:
            # Extract audio if needed
            audio_path = self._extract_audio(media_path) if extract_audio else media_path
            
            # Get full transcription with timestamps
            transcription = self.audio_processor.transcribe_audio(audio_path)
            
            if "error" in transcription:
                logger.error(f"Transcription error: {transcription['error']}")
                return []
            
            # Analyze content for interesting moments
            content_analysis = self.audio_processor.analyze_audio_content(audio_path)
            
            # Find keyword matches if keywords provided
            keyword_matches = {}
            if keywords and len(keywords) > 0:
                keyword_matches = self.audio_processor.detect_keywords(audio_path, keywords)
            
            # Generate clip suggestions
            suggestions = self._generate_suggestions(
                transcription, 
                content_analysis,
                keyword_matches,
                max_suggestions
            )
            
            # Clean up temporary files
            if extract_audio and audio_path != media_path:
                try:
                    os.remove(audio_path)
                except Exception as e:
                    logger.warning(f"Failed to remove temporary audio file: {str(e)}")
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error suggesting clips: {str(e)}")
            return []
    
    def _extract_audio(self, media_path: str) -> str:
        """
        Extract audio from a video file.
        
        Args:
            media_path: Path to the video file
            
        Returns:
            Path to the extracted audio file
        """
        try:
            # Create a temporary file for the extracted audio
            temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_audio.close()
            
            # Extract audio using ffmpeg
            cmd = [
                "ffmpeg",
                "-i", media_path,
                "-vn",                  # No video
                "-acodec", "pcm_s16le", # PCM format
                "-ar", "16000",         # 16kHz sample rate
                "-ac", "1",             # Mono
                "-y",                   # Overwrite output file
                temp_audio.name
            ]
            
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if process.returncode != 0:
                logger.error(f"FFmpeg error: {process.stderr}")
                raise RuntimeError(f"Failed to extract audio: {process.stderr}")
            
            return temp_audio.name
            
        except Exception as e:
            logger.error(f"Error extracting audio: {str(e)}")
            raise
    
    def _generate_suggestions(
        self,
        transcription: Dict[str, Any],
        content_analysis: Dict[str, Any],
        keyword_matches: Dict[str, List[Dict[str, Any]]],
        max_suggestions: int
    ) -> List[Dict[str, Any]]:
        """
        Generate clip suggestions based on analysis results.
        
        Args:
            transcription: Transcription results with timestamps
            content_analysis: Content analysis results
            keyword_matches: Keyword match results
            max_suggestions: Maximum number of suggestions to return
            
        Returns:
            List of clip suggestion dictionaries
        """
        suggestions = []
        segments = transcription.get("segments", [])
        
        if not segments:
            logger.warning("No segments found in transcription")
            return []
        
        # 1. Find segments with keywords (highest priority)
        keyword_clips = self._find_keyword_clips(segments, keyword_matches)
        suggestions.extend(keyword_clips)
        
        # 2. Find segments with emotional content
        emotion_clips = self._find_emotion_clips(segments, content_analysis)
        for clip in emotion_clips:
            if clip not in suggestions:
                suggestions.append(clip)
        
        # 3. Find coherent segments within duration limits
        if len(suggestions) < max_suggestions:
            content_clips = self._find_content_clips(segments)
            for clip in content_clips:
                if clip not in suggestions and len(suggestions) < max_suggestions:
                    suggestions.append(clip)
        
        # Limit to max_suggestions
        suggestions = suggestions[:max_suggestions]
        
        # Sort by start time
        suggestions.sort(key=lambda x: x["start"])
        
        return suggestions
    
    def _find_keyword_clips(
        self, 
        segments: List[Dict[str, Any]],
        keyword_matches: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Find clip suggestions based on keyword matches.
        
        Args:
            segments: Transcription segments
            keyword_matches: Keyword match results
            
        Returns:
            List of clip suggestions
        """
        suggestions = []
        
        # Process each keyword
        for keyword, occurrences in keyword_matches.items():
            if not occurrences:
                continue
                
            for occurrence in occurrences:
                # Find start and end segments
                start_time = occurrence["start"]
                end_time = occurrence["end"]
                
                # Extend clip to include context
                extended_start = max(0, start_time - 2.0)  # 2 seconds before keyword
                extended_end = end_time + 3.0              # 3 seconds after keyword
                
                # Ensure minimum duration
                if extended_end - extended_start < self.min_clip_duration:
                    extended_end = extended_start + self.min_clip_duration
                
                # Cap to maximum duration
                if extended_end - extended_start > self.max_clip_duration:
                    # Try to center the keyword in the clip
                    middle = (start_time + end_time) / 2
                    half_max = self.max_clip_duration / 2
                    extended_start = max(0, middle - half_max)
                    extended_end = extended_start + self.max_clip_duration
                
                # Create suggestion
                suggestion = {
                    "start": extended_start,
                    "end": extended_end,
                    "text": occurrence["text"],
                    "reason": f"Contains keyword '{keyword}'",
                    "score": 0.9,  # High score for keyword matches
                    "type": "keyword"
                }
                
                suggestions.append(suggestion)
        
        return suggestions
    
    def _find_emotion_clips(
        self, 
        segments: List[Dict[str, Any]],
        content_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Find clip suggestions based on emotional content.
        
        Args:
            segments: Transcription segments
            content_analysis: Content analysis results
            
        Returns:
            List of clip suggestions
        """
        suggestions = []
        
        # Look for emotion indicators in the analysis
        emotion_indicators = [
            "excitement", "excited", "laugh", "laughing", "cheer", "cheering",
            "surprise", "surprised", "shock", "shocked", "angry", "anger",
            "sad", "sadness", "emotional", "happy", "happiness", "fear",
            "scared", "cry", "crying", "yell", "yelling", "shout", "shouting"
        ]
        
        # Find segments with emotional content
        for i, segment in enumerate(segments):
            text = segment["text"].lower()
            
            # Check if any emotion indicators are in the text
            emotion_found = any(indicator in text for indicator in emotion_indicators)
            
            if emotion_found:
                # Find a good clip boundary
                start_idx = max(0, i - 1)  # Include previous segment
                end_idx = min(len(segments) - 1, i + 1)  # Include next segment
                
                start_time = segments[start_idx]["start"]
                end_time = segments[end_idx]["end"]
                
                # Ensure minimum duration
                if end_time - start_time < self.min_clip_duration:
                    end_time = start_time + self.min_clip_duration
                
                # Cap to maximum duration
                if end_time - start_time > self.max_clip_duration:
                    middle = segment["start"] + (segment["end"] - segment["start"]) / 2
                    half_max = self.max_clip_duration / 2
                    start_time = max(0, middle - half_max)
                    end_time = start_time + self.max_clip_duration
                
                # Create suggestion
                suggestion = {
                    "start": start_time,
                    "end": end_time,
                    "text": segment["text"],
                    "reason": "Emotional content detected",
                    "score": 0.8,  # High score for emotional content
                    "type": "emotion"
                }
                
                suggestions.append(suggestion)
        
        return suggestions
    
    def _find_content_clips(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Find clip suggestions based on content coherence.
        
        Args:
            segments: Transcription segments
            
        Returns:
            List of clip suggestions
        """
        suggestions = []
        
        # Find coherent segments that fit within duration limits
        current_start = 0
        current_end = 0
        current_text = []
        
        for segment in segments:
            # If adding this segment exceeds max duration, create a suggestion
            if segment["end"] - current_start > self.max_clip_duration:
                if current_end - current_start >= self.min_clip_duration:
                    suggestion = {
                        "start": current_start,
                        "end": current_end,
                        "text": " ".join(current_text),
                        "reason": "Coherent content",
                        "score": 0.5,  # Medium score for coherent content
                        "type": "content"
                    }
                    suggestions.append(suggestion)
                
                # Start a new potential clip
                current_start = segment["start"]
                current_text = []
            
            # Add segment to current clip
            current_end = segment["end"]
            current_text.append(segment["text"])
        
        # Add the last clip if it meets minimum duration
        if current_end - current_start >= self.min_clip_duration:
            suggestion = {
                "start": current_start,
                "end": current_end,
                "text": " ".join(current_text),
                "reason": "Coherent content",
                "score": 0.5,
                "type": "content"
            }
            suggestions.append(suggestion)
        
        return suggestions 