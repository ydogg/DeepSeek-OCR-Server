"""
Configuration loader for DeepSeek OCR
Loads configuration from YAML files
"""

import os
import yaml
from typing import Dict, Any


def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from a YAML file"""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


class ConfigDict(dict):
    """A dictionary subclass that allows both attribute-style and key-style access"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Convert nested dictionaries to ConfigDict
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = ConfigDict(value)
    
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")
    
    def __setattr__(self, key, value):
        self[key] = value


# Load configurations
COMMON_CONFIG = ConfigDict(load_config(os.path.join(os.path.dirname(__file__), 'config.yaml')))
SERVER_CONFIG = ConfigDict(load_config(os.path.join(os.path.dirname(__file__), 'server_config.yaml')))
BATCH_CONFIG = ConfigDict(load_config(os.path.join(os.path.dirname(__file__), 'batch_config.yaml')))