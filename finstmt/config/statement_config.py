from dataclasses import dataclass
from typing import List
import os
from pathlib import Path

from finstmt.items.config import ItemConfig
from finstmt.config.config_loader import load_yaml_config

@dataclass
class StatementConfig:
    key: str
    display_name: str
    items_config_list: List[ItemConfig]

def get_default_config_path() -> str:
    """Get the path to the default YAML config file"""
    return str(Path(__file__).parent / 'statement_configs.yaml')

def load_statement_configs(config_path: str = None) -> List[StatementConfig]:
    """
    Load statement configurations, using default if no path specified
    
    :param config_path: Optional path to YAML config file
    :return: List of StatementConfig objects
    """
    if config_path is None:
        config_path = get_default_config_path()
        
    return load_yaml_config(config_path)

# Load the default configurations
STATEMENT_CONFIGS = load_statement_configs()

# Keep individual statement configs for backwards compatibility
BALANCE_SHEET_CONFIG = next(cfg for cfg in STATEMENT_CONFIGS if cfg.key == 'bs')
INCOME_STATEMENT_CONFIG = next(cfg for cfg in STATEMENT_CONFIGS if cfg.key == 'inc')
METRICS_STATEMENT_CONFIG = next(cfg for cfg in STATEMENT_CONFIGS if cfg.key == 'metrics')
