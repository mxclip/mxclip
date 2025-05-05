"""
Stream URL resolver for MX Clipping.

This module resolves user-friendly URLs (like Twitch, YouTube) into direct streaming URLs
that can be used for recording and processing.
"""

import re
import subprocess
import logging
import shutil
from typing import Optional, Tuple, Dict
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class StreamResolver:
    """
    Resolves platform-specific URLs into direct streaming URLs.
    
    Supports:
    - Twitch (using streamlink)
    - YouTube (using yt-dlp)
    - Direct stream URLs (m3u8, rtmp, etc.)
    """
    
    PLATFORM_PATTERNS = {
        'twitch': r'(twitch\.tv)',
        'youtube': r'(youtube\.com|youtu\.be)',
        'bilibili': r'(bilibili\.com|live\.bilibili\.com)',
    }
    
    def __init__(self):
        """Initialize the stream resolver."""
        self._check_dependencies()
    
    def _check_dependencies(self) -> Dict[str, bool]:
        """
        Check if required external tools are available.
        
        Returns:
            Dictionary with tool availability status
        """
        dependencies = {
            'streamlink': shutil.which('streamlink') is not None,
            'yt-dlp': shutil.which('yt-dlp') is not None
        }
        
        # Log missing dependencies
        for tool, available in dependencies.items():
            if not available:
                logger.warning(f"External tool '{tool}' not found. Some stream resolution may fail.")
        
        return dependencies
    
    def detect_platform(self, url: str) -> str:
        """
        Detect streaming platform from URL.
        
        Args:
            url: URL to analyze
            
        Returns:
            Platform name or 'direct' if no platform detected
        """
        # Parse URL for better matching
        parsed = urlparse(url)
        domain = parsed.netloc + parsed.path
        
        # Check against known platform patterns
        for platform, pattern in self.PLATFORM_PATTERNS.items():
            if re.search(pattern, domain, re.IGNORECASE):
                logger.info(f"Detected {platform} URL: {url}")
                return platform
        
        # If looks like a direct stream URL
        if url.startswith(('rtmp://', 'rtmps://', 'http://', 'https://')) and any(
            ext in url for ext in ['.m3u8', '.mpd', '.flv']):
            logger.info(f"Detected direct stream URL: {url}")
            return 'direct'
        
        logger.info(f"Unknown platform URL: {url} (treating as direct)")
        return 'direct'
    
    def resolve_stream_url(self, url: str, quality: str = 'best') -> Tuple[Optional[str], Optional[str]]:
        """
        Resolve direct streaming URL from platform URL.
        
        Args:
            url: Platform URL or direct stream URL
            quality: Desired stream quality (best, worst, 720p, etc.)
            
        Returns:
            Tuple of (resolved_url, error_message)
        """
        platform = self.detect_platform(url)
        
        try:
            if platform == 'twitch':
                return self._resolve_twitch(url, quality), None
            elif platform == 'youtube':
                return self._resolve_youtube(url, quality), None
            elif platform == 'bilibili':
                # TODO: implement Bilibili resolution
                return url, "Bilibili resolution not implemented yet"
            else:
                # Direct URL, return as is
                return url, None
        except Exception as e:
            error_msg = f"Error resolving stream URL: {str(e)}"
            logger.error(error_msg)
            return None, error_msg
    
    def _resolve_twitch(self, url: str, quality: str = 'best') -> Optional[str]:
        """
        Resolve Twitch URL using streamlink.
        
        Args:
            url: Twitch URL
            quality: Stream quality
            
        Returns:
            Direct stream URL or None if resolution fails
        """
        try:
            result = subprocess.run(
                ['streamlink', url, quality, '--stream-url'],
                capture_output=True, 
                text=True,
                check=True
            )
            stream_url = result.stdout.strip()
            
            if not stream_url:
                logger.error(f"Failed to resolve Twitch URL: {url}")
                return None
                
            logger.info(f"Resolved Twitch URL: {url} -> [stream URL hidden for brevity]")
            return stream_url
        except subprocess.CalledProcessError as e:
            logger.error(f"streamlink error: {e.stderr}")
            if "error: No playable streams found" in e.stderr:
                logger.error(f"No active stream found at {url}")
            return None
    
    def _resolve_youtube(self, url: str, quality: str = 'best') -> Optional[str]:
        """
        Resolve YouTube URL using yt-dlp.
        
        Args:
            url: YouTube URL
            quality: Stream quality
            
        Returns:
            Direct stream URL or None if resolution fails
        """
        try:
            result = subprocess.run(
                ['yt-dlp', '-g', '-f', quality, url],
                capture_output=True, 
                text=True,
                check=True
            )
            stream_url = result.stdout.strip()
            
            if not stream_url:
                logger.error(f"Failed to resolve YouTube URL: {url}")
                return None
                
            logger.info(f"Resolved YouTube URL: {url} -> [stream URL hidden for brevity]")
            return stream_url
        except subprocess.CalledProcessError as e:
            logger.error(f"yt-dlp error: {e.stderr}")
            return None 