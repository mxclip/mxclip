"""
Configuration loader for MX Clipping.

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
    Loads and validates user configurations for MX Clipping.
    
    Configurations include:
    - User ID
    - Keywords to monitor
    - Repetition detection settings
    - Chat activity monitoring settings
    """
    
    DEFAULT_CONFIG = {
        "user_id": "default_user",
        "keywords": ["wow", "amazing", "let's go", "oh my god", "no way"],
        "enable_repeat_check": True,
        "repeat_window_seconds": 10,
        "repeat_threshold": 2,
        "enable_chat_check": True,
        "chat_activity_threshold": 15
    }
    
    def __init__(self, config_dir: str = "configs"):
        """
        Initialize the config loader.
        
        Args:
            config_dir: Directory containing user configuration files
        """
        self.config_dir = config_dir
        
        # Create config directory if it doesn't exist
        os.makedirs(config_dir, exist_ok=True)
        
        # Track loaded configs
        self.loaded_configs = {}
    
    def load_config(self, user_id: str) -> Dict[str, Any]:
        """
        Load configuration for a specific user.
        
        If the user's config file doesn't exist, creates it with default values.
        
        Args:
            user_id: The user ID to load configuration for
        
        Returns:
            The user's configuration
        """
        config_path = os.path.join(self.config_dir, f"{user_id}_config.json")
        
        # If config already loaded, return it
        if user_id in self.loaded_configs:
            return self.loaded_configs[user_id]
        
        # If config file exists, load it
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration for user {user_id}")
                
                # Validate and fill in missing fields with defaults
                config = self._validate_config(config, user_id)
                
            except Exception as e:
                logger.error(f"Error loading config for user {user_id}: {str(e)}")
                logger.info(f"Using default configuration for user {user_id}")
                config = self._create_default_config(user_id)
        else:
            # Create default config
            logger.info(f"No configuration found for user {user_id}, creating default")
            config = self._create_default_config(user_id)
            self._save_config(config, user_id)
        
        # Store loaded config
        self.loaded_configs[user_id] = config
        return config
    
    def update_config(self, user_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a user's configuration.
        
        Args:
            user_id: The user ID to update configuration for
            updates: Dictionary of configuration updates
        
        Returns:
            The updated configuration
        """
        # Load current config
        config = self.load_config(user_id)
        
        # Apply updates
        for key, value in updates.items():
            if key in config:
                config[key] = value
        
        # Validate updated config
        config = self._validate_config(config, user_id)
        
        # Save updated config
        self._save_config(config, user_id)
        
        # Update loaded config
        self.loaded_configs[user_id] = config
        
        logger.info(f"Updated configuration for user {user_id}")
        return config
    
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
                if filename.endswith("_config.json"):
                    user_id = filename.replace("_config.json", "")
                    user_ids.append(user_id)
        
        return user_ids
    
    def _create_default_config(self, user_id: str) -> Dict[str, Any]:
        """
        Create a default configuration for a user.
        
        Args:
            user_id: The user ID to create configuration for
        
        Returns:
            The default configuration
        """
        config = self.DEFAULT_CONFIG.copy()
        config["user_id"] = user_id
        return config
    
    def _validate_config(self, config: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """
        Validate a configuration, filling in missing fields with defaults.
        
        Args:
            config: The configuration to validate
            user_id: The user ID the configuration belongs to
        
        Returns:
            The validated configuration
        """
        default_config = self._create_default_config(user_id)
        validated_config = {}
        
        # Ensure all required fields are present
        for key, default_value in default_config.items():
            if key in config:
                # Validate value type
                if isinstance(config[key], type(default_value)):
                    validated_config[key] = config[key]
                else:
                    logger.warning(f"Invalid type for {key} in user {user_id}'s config, using default")
                    validated_config[key] = default_value
            else:
                # Field missing, use default
                logger.warning(f"Missing {key} in user {user_id}'s config, using default")
                validated_config[key] = default_value
        
        return validated_config
    
    def _save_config(self, config: Dict[str, Any], user_id: str) -> None:
        """
        Save a configuration to disk.
        
        Args:
            config: The configuration to save
            user_id: The user ID the configuration belongs to
        """
        config_path = os.path.join(self.config_dir, f"{user_id}_config.json")
        
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Saved configuration for user {user_id}")
        except Exception as e:
            logger.error(f"Error saving config for user {user_id}: {str(e)}") 