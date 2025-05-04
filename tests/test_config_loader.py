"""
Tests for the ConfigLoader module.
"""

import os
import json
import shutil
import tempfile
import pytest
from mxclip.config_loader import ConfigLoader

class TestConfigLoader:
    """Test suite for ConfigLoader."""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary directory for config files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_create_default_config(self, temp_config_dir):
        """Test loading a non-existent config creates a default one."""
        # Create config loader with temp directory
        config_loader = ConfigLoader(config_dir=temp_config_dir)
        
        # Load config for non-existent user
        user_id = "test_user"
        config = config_loader.load_config(user_id)
        
        # Check config has expected values
        assert config["user_id"] == user_id
        assert "keywords" in config
        assert "enable_repeat_check" in config
        
        # Check config file was created
        config_path = os.path.join(temp_config_dir, f"{user_id}_config.json")
        assert os.path.exists(config_path)
        
        # Load the config file directly and verify contents
        with open(config_path, 'r') as f:
            saved_config = json.load(f)
        assert saved_config["user_id"] == user_id
    
    def test_load_existing_config(self, temp_config_dir):
        """Test loading an existing config file."""
        # Create a config file
        user_id = "existing_user"
        config_path = os.path.join(temp_config_dir, f"{user_id}_config.json")
        
        test_config = {
            "user_id": user_id,
            "keywords": ["test1", "test2"],
            "enable_repeat_check": False,
            "repeat_window_seconds": 5,
            "repeat_threshold": 3,
            "enable_chat_check": True,
            "chat_activity_threshold": 10
        }
        
        with open(config_path, 'w') as f:
            json.dump(test_config, f)
        
        # Create config loader and load the config
        config_loader = ConfigLoader(config_dir=temp_config_dir)
        loaded_config = config_loader.load_config(user_id)
        
        # Check loaded config matches the test config
        assert loaded_config["user_id"] == user_id
        assert loaded_config["keywords"] == ["test1", "test2"]
        assert loaded_config["enable_repeat_check"] is False
        assert loaded_config["repeat_window_seconds"] == 5
    
    def test_update_config(self, temp_config_dir):
        """Test updating a user's configuration."""
        # Create config loader with temp directory
        config_loader = ConfigLoader(config_dir=temp_config_dir)
        
        # Load default config for a user
        user_id = "update_test_user"
        original_config = config_loader.load_config(user_id)
        
        # Update some settings
        updates = {
            "keywords": ["updated1", "updated2"],
            "repeat_threshold": 5
        }
        
        updated_config = config_loader.update_config(user_id, updates)
        
        # Check updates were applied
        assert updated_config["keywords"] == ["updated1", "updated2"]
        assert updated_config["repeat_threshold"] == 5
        
        # Check other settings remained unchanged
        assert updated_config["enable_repeat_check"] == original_config["enable_repeat_check"]
        
        # Load the config again to verify changes were saved
        reloaded_config = config_loader.load_config(user_id)
        assert reloaded_config["keywords"] == ["updated1", "updated2"]
    
    def test_validate_config(self, temp_config_dir):
        """Test validation of configuration with missing or invalid fields."""
        # Create a config file with missing fields
        user_id = "invalid_user"
        config_path = os.path.join(temp_config_dir, f"{user_id}_config.json")
        
        invalid_config = {
            "user_id": user_id,
            "keywords": ["test"],
            # Missing other fields
        }
        
        with open(config_path, 'w') as f:
            json.dump(invalid_config, f)
        
        # Create config loader and load the config
        config_loader = ConfigLoader(config_dir=temp_config_dir)
        loaded_config = config_loader.load_config(user_id)
        
        # Check missing fields were filled with defaults
        assert "enable_repeat_check" in loaded_config
        assert "repeat_window_seconds" in loaded_config
        assert "repeat_threshold" in loaded_config
        
        # Create a config with invalid field types
        user_id = "type_invalid_user"
        config_path = os.path.join(temp_config_dir, f"{user_id}_config.json")
        
        type_invalid_config = {
            "user_id": user_id,
            "keywords": "not_a_list",  # Should be a list
            "repeat_threshold": "5"     # Should be an integer
        }
        
        with open(config_path, 'w') as f:
            json.dump(type_invalid_config, f)
        
        # Load and validate the config
        loaded_config = config_loader.load_config(user_id)
        
        # Check invalid fields were replaced with defaults
        assert isinstance(loaded_config["keywords"], list)
        assert isinstance(loaded_config["repeat_threshold"], int)
    
    def test_get_all_user_ids(self, temp_config_dir):
        """Test retrieving all user IDs with config files."""
        # Create config loader with temp directory
        config_loader = ConfigLoader(config_dir=temp_config_dir)
        
        # Create configs for several users
        users = ["user1", "user2", "user3"]
        for user_id in users:
            config_loader.load_config(user_id)
        
        # Get all user IDs
        all_users = config_loader.get_all_user_ids()
        
        # Check all users are included
        for user_id in users:
            assert user_id in all_users
        
        # Check no extra users are included
        assert len(all_users) == len(users) 