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
        config_path = os.path.join(self.config_dir, f"{user_id}_config.json")
        
        # Start with default config
        config = self.default_config.copy()
        
        # Add user_id to config
        config["user_id"] = user_id
        
        # Try to load user-specific config
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                
                # Update defaults with user-specific values
                config.update(user_config)
                
                # Ensure user_id is correctly set
                config["user_id"] = user_id
                
                logger.info(f"Loaded configuration for user {user_id}")
            except Exception as e:
                logger.error(f"Error loading config for user {user_id}: {str(e)}")
        else:
            logger.info(f"No configuration found for user {user_id}, using defaults")
            
            # Create a default config file for the user
            self.save_config(user_id, config)
        
        # Validate configuration
        self._validate_config(config)
        
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
        config_path = os.path.join(self.config_dir, f"{user_id}_config.json")
        
        # Ensure user_id is in the config
        config["user_id"] = user_id
        
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
        
        # Ensure user_id is not overwritten
        config["user_id"] = user_id
        
        # Validate updates
        self._validate_config(config)
        
        # Save updated config
        self.save_config(user_id, config)
        
        return config
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration and fix any invalid values.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Validated configuration dictionary
        """
        # Ensure required fields with correct types
        if not isinstance(config.get("keywords", []), list):
            config["keywords"] = self.default_config["keywords"]
            
        if not isinstance(config.get("enable_repeat_check", True), bool):
            config["enable_repeat_check"] = self.default_config["enable_repeat_check"]
            
        if not isinstance(config.get("repeat_window_seconds", 10.0), (int, float)):
            config["repeat_window_seconds"] = self.default_config["repeat_window_seconds"]
            
        if not isinstance(config.get("repeat_threshold", 2), int):
            config["repeat_threshold"] = self.default_config["repeat_threshold"]
            
        if not isinstance(config.get("enable_chat_check", True), bool):
            config["enable_chat_check"] = self.default_config["enable_chat_check"]
            
        if not isinstance(config.get("chat_activity_threshold", 5), int):
            config["chat_activity_threshold"] = self.default_config["chat_activity_threshold"]
            
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
                if filename.endswith("_config.json"):
                    user_id = filename.replace("_config.json", "")
                    user_ids.append(user_id)
        
        return user_ids 