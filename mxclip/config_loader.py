"""
Config Loader for MX Clipping.

This module handles loading and validating user configurations
for keyword detection, repetition detection, and chat activity monitoring.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class ConfigLoader:
    """
    Loads and manages user configurations for the clipping service.
    """
    
    def __init__(self, config_dir: str = "configs"):
        """
        Initialize the config loader.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = config_dir
        os.makedirs(config_dir, exist_ok=True)
        
        # Default configuration values
        self.default_config = {
            "keywords": ["awesome", "amazing", "wow", "cool", "nice"],
            "enable_repeat_check": True,
            "repeat_window_seconds": 10.0,
            "repeat_threshold": 2,
            "enable_chat_check": True,
            "chat_activity_threshold": 5
        }
        
        logger.info(f"Initialized ConfigLoader with config directory: {config_dir}")
    
    def load_config(self, user_id: str) -> Dict[str, Any]:
        """
        Load configuration for a specific user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Configuration dictionary with default values filled in
        """
        config_path = os.path.join(self.config_dir, f"{user_id}.json")
        
        # Start with default config
        config = self.default_config.copy()
        
        # Try to load user-specific config
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                
                # Update defaults with user-specific values
                config.update(user_config)
                logger.info(f"Loaded configuration for user {user_id}")
            except Exception as e:
                logger.error(f"Error loading config for user {user_id}: {str(e)}")
        else:
            logger.info(f"No configuration found for user {user_id}, using defaults")
            
            # Create a default config file for the user
            self.save_config(user_id, config)
        
        return config
    
    def save_config(self, user_id: str, config: Dict[str, Any]) -> bool:
        """
        Save configuration for a specific user.
        
        Args:
            user_id: User identifier
            config: Configuration dictionary
            
        Returns:
            True if saved successfully, False otherwise
        """
        config_path = os.path.join(self.config_dir, f"{user_id}.json")
        
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Saved configuration for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving config for user {user_id}: {str(e)}")
            return False
    
    def update_config(self, user_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update specific configuration values for a user.
        
        Args:
            user_id: User identifier
            updates: Dictionary of configuration updates
            
        Returns:
            Updated configuration dictionary
        """
        # Load current config
        config = self.load_config(user_id)
        
        # Apply updates
        config.update(updates)
        
        # Save updated config
        self.save_config(user_id, config)
        
        return config
    
    def get_value(self, user_id: str, key: str, default: Any = None) -> Any:
        """
        Get a specific configuration value for a user.
        
        Args:
            user_id: User identifier
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        config = self.load_config(user_id)
        return config.get(key, default)
    
    def get_all_user_ids(self) -> List[str]:
        """
        Get a list of all user IDs with configuration files.
        
        Returns:
            List of user IDs
        """
        user_ids = []
        
        # Look for config files in the config directory
        if os.path.exists(self.config_dir):
            for filename in os.listdir(self.config_dir):
                if filename.endswith(".json"):
                    user_id = filename.replace(".json", "")
                    user_ids.append(user_id)
        
        return user_ids 