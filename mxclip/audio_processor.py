"""
Audio processor using Kimi-Audio for MX Clipping.

This module integrates Kimi-Audio capabilities to enhance audio processing,
including advanced transcription, audio understanding, and audio classification.
"""

import os
import logging
import tempfile
from typing import Dict, List, Optional, Tuple, Any
import subprocess
import torch
import numpy as np

logger = logging.getLogger(__name__)

class KimiAudioProcessor:
    """
    Processor that uses Kimi-Audio models for enhanced audio processing.
    
    Features:
    - High-quality audio transcription
    - Audio content understanding and classification
    - Audio event detection
    - Emotional tone analysis
    """
    
    def __init__(
        self, 
        model_name: str = "kimi-audio-7b-instruct",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_flash_attn: bool = True,
        audio_chunk_size_sec: int = 30,
    ):
        """
        Initialize the Kimi Audio processor.
        
        Args:
            model_name: Name of the Kimi-Audio model to use
            device: Device to run the model on ("cuda" or "cpu")
            use_flash_attn: Whether to use flash attention for faster processing
            audio_chunk_size_sec: Size of audio chunks for processing in seconds
        """
        self.model_name = model_name
        self.device = device
        self.use_flash_attn = use_flash_attn
        self.audio_chunk_size_sec = audio_chunk_size_sec
        self.model = None
        self.processor = None
        
        try:
            self._load_model()
            logger.info(f"Kimi-Audio model '{model_name}' loaded successfully on {device}")
        except Exception as e:
            logger.error(f"Failed to load Kimi-Audio model: {str(e)}")
            raise
    
    def _load_model(self) -> None:
        """Load the Kimi-Audio model and processor."""
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM
            
            logger.info(f"Loading Kimi-Audio model '{self.model_name}'...")
            
            # Load the processor and model
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device,
                use_flash_attention_2=self.use_flash_attn if self.device == "cuda" else False,
            )
            
            logger.info(f"Kimi-Audio model loaded successfully")
        except ImportError as e:
            logger.error(f"Required libraries not installed: {str(e)}")
            logger.error("Please install transformers and related dependencies")
            raise
        except Exception as e:
            logger.error(f"Error loading Kimi-Audio model: {str(e)}")
            raise
    
    def transcribe_audio(self, audio_path: str, language: str = "auto") -> Dict[str, Any]:
        """
        Transcribe audio file using Kimi-Audio's advanced transcription capabilities.
        
        Args:
            audio_path: Path to the audio file
            language: Language code or "auto" for auto-detection
            
        Returns:
            Dictionary containing transcription results with timestamps
        """
        try:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Process audio with Kimi-Audio model
            inputs = self.processor(
                audio=audio_path,
                return_tensors="pt",
                sampling_rate=16000,
                add_special_tokens=True,
            ).to(self.device)
            
            # Generate transcription
            prompt = f"Please transcribe the audio with timestamps."
            input_ids = self.processor.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    audio_values=inputs.audio_values,
                    audio_padding_mask=inputs.audio_padding_mask,
                    max_new_tokens=4096,
                    do_sample=False,
                )
            
            # Process the transcription output
            transcription = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Parse timestamps and text
            transcript_segments = self._parse_transcription(transcription)
            
            return {
                "text": transcription,
                "segments": transcript_segments,
                "language": language,
            }
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            return {"error": str(e)}
    
    def _parse_transcription(self, transcription: str) -> List[Dict[str, Any]]:
        """
        Parse the transcription output to extract timestamps and text segments.
        
        Args:
            transcription: Raw transcription text with timestamps
            
        Returns:
            List of segment dictionaries with start_time, end_time, and text
        """
        import re
        
        # Pattern for timestamps like [00:01:23 - 00:01:45]
        pattern = r'\[(\d{2}:\d{2}:\d{2}) - (\d{2}:\d{2}:\d{2})\](.*?)(?=\[\d{2}:\d{2}:\d{2} - \d{2}:\d{2}:\d{2}\]|$)'
        
        segments = []
        for match in re.finditer(pattern, transcription, re.DOTALL):
            start_time_str, end_time_str, text = match.groups()
            
            # Convert time strings to seconds
            start_seconds = self._time_str_to_seconds(start_time_str)
            end_seconds = self._time_str_to_seconds(end_time_str)
            
            segments.append({
                "start": start_seconds,
                "end": end_seconds,
                "text": text.strip()
            })
        
        # If no timestamps found, return the whole text as one segment
        if not segments:
            segments = [{
                "start": 0,
                "end": 0,  # Unknown duration
                "text": transcription.strip()
            }]
        
        return segments
    
    def _time_str_to_seconds(self, time_str: str) -> float:
        """
        Convert a time string (HH:MM:SS) to seconds.
        
        Args:
            time_str: Time string in format HH:MM:SS
            
        Returns:
            Time in seconds
        """
        h, m, s = map(int, time_str.split(':'))
        return h * 3600 + m * 60 + s
    
    def analyze_audio_content(self, audio_path: str) -> Dict[str, Any]:
        """
        Analyze audio content for events, tone, and other characteristics.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary with analysis results
        """
        try:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Process audio with Kimi-Audio model
            inputs = self.processor(
                audio=audio_path,
                return_tensors="pt",
                sampling_rate=16000,
                add_special_tokens=True,
            ).to(self.device)
            
            # Generate analysis
            prompt = f"Please analyze this audio and describe the content, emotional tone, and any notable events or sounds."
            input_ids = self.processor.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    audio_values=inputs.audio_values,
                    audio_padding_mask=inputs.audio_padding_mask,
                    max_new_tokens=1024,
                    do_sample=False,
                )
            
            # Process the analysis output
            analysis = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extract structured information from the analysis text
            result = self._parse_analysis(analysis)
            
            return result
        except Exception as e:
            logger.error(f"Error analyzing audio content: {str(e)}")
            return {"error": str(e)}
    
    def _parse_analysis(self, analysis: str) -> Dict[str, Any]:
        """
        Parse the analysis output to extract structured information.
        
        Args:
            analysis: Raw analysis text
            
        Returns:
            Dictionary with structured analysis results
        """
        # Simple parsing - in a real implementation this would use more
        # sophisticated NLP to extract structured information
        result = {
            "full_analysis": analysis,
            "summary": analysis.split('\n\n')[0] if '\n\n' in analysis else analysis,
        }
        
        # Try to extract emotional tone
        if "emotion:" in analysis.lower() or "tone:" in analysis.lower():
            import re
            emotion_match = re.search(r'emotion[s]?:(.+?)(?:\.|$|\n)', analysis, re.IGNORECASE)
            tone_match = re.search(r'tone:(.+?)(?:\.|$|\n)', analysis, re.IGNORECASE)
            
            if emotion_match:
                result["emotion"] = emotion_match.group(1).strip()
            if tone_match:
                result["tone"] = tone_match.group(1).strip()
        
        return result
    
    def detect_keywords(self, audio_path: str, keywords: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Detect specific keywords in audio content.
        
        Args:
            audio_path: Path to the audio file
            keywords: List of keywords to detect
            
        Returns:
            Dictionary mapping keywords to lists of occurrences with timestamps
        """
        # First get the full transcription with timestamps
        transcription = self.transcribe_audio(audio_path)
        
        if "error" in transcription:
            return {"error": transcription["error"]}
        
        results = {}
        
        # Check each segment for keywords
        for keyword in keywords:
            keyword_lower = keyword.lower()
            occurrences = []
            
            for segment in transcription.get("segments", []):
                text_lower = segment["text"].lower()
                
                if keyword_lower in text_lower:
                    occurrences.append({
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": segment["text"]
                    })
            
            results[keyword] = occurrences
        
        return results
    
    def generate_audio_caption(self, audio_path: str) -> str:
        """
        Generate a concise caption describing the audio content.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Caption string
        """
        try:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Process audio with Kimi-Audio model
            inputs = self.processor(
                audio=audio_path,
                return_tensors="pt",
                sampling_rate=16000,
                add_special_tokens=True,
            ).to(self.device)
            
            # Generate caption
            prompt = f"Please provide a short, concise caption for this audio."
            input_ids = self.processor.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    audio_values=inputs.audio_values,
                    audio_padding_mask=inputs.audio_padding_mask,
                    max_new_tokens=100,
                    do_sample=False,
                )
            
            # Process the caption output
            caption = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Remove any instructions or prefixes that might be in the response
            caption = caption.replace(prompt, "").strip()
            if caption.startswith("Caption:"):
                caption = caption[8:].strip()
            
            return caption
        except Exception as e:
            logger.error(f"Error generating audio caption: {str(e)}")
            return f"Error: {str(e)}" 