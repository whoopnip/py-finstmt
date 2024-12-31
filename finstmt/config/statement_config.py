from dataclasses import dataclass
from typing import List
import os
from pathlib import Path

from finstmt.findata.item_config import ItemConfig
from finstmt.findata.item_forecast_config import ForecastItemConfig
from finstmt.forecast.config import ForecastConfig
from finstmt.config.config_loader import load_yaml_config

@dataclass
class StatementConfig:
    key: str
    display_name: str
    items_config_list: List[ItemConfig]

def get_default_config_path() -> str:
    """Get the path to the default YAML config file"""
    return str(Path(__file__).parent / 'statement_config.yaml')

def load_statement_configs(config_path: str = None) -> List[StatementConfig]:
    """
    Load statement configurations, using default if no path specified
    
    :param config_path: Optional path to YAML config file
    :return: List of StatementConfig objects
    """
    if config_path is None:
        config_path = get_default_config_path()
        
    config_data = load_yaml_config(config_path)
    statement_configs = []
    
    for stmt_config in config_data['statements']:
        items_config = []
        for item in stmt_config['items']:
            forecast_config = ForecastItemConfig()
            if 'forecast' in item:
                forecast_config = ForecastItemConfig(**item['forecast'])
            
            items_config.append(ItemConfig(
                key=item['key'],
                display_name=item['display_name'],
                extract_names=item.get('extract_names', None),
                expr_str=item.get('expr_str'),
                force_positive=item.get('force_positive', True),
                forecast_config=forecast_config,
                display_verbosity=item.get('display_verbosity', 1)
            ))
            
        statement_configs.append(StatementConfig(
            key=stmt_config['key'],
            display_name=stmt_config['display_name'],
            items_config_list=items_config
        ))
    
    return statement_configs

# Load the default configurations
STATEMENT_CONFIGS = load_statement_configs()

# Keep individual statement configs for backwards compatibility
BALANCE_SHEET_CONFIG = next(cfg for cfg in STATEMENT_CONFIGS if cfg.key == 'bs')
INCOME_STATEMENT_CONFIG = next(cfg for cfg in STATEMENT_CONFIGS if cfg.key == 'inc')
METRICS_STATEMENT_CONFIG = next(cfg for cfg in STATEMENT_CONFIGS if cfg.key == 'metrics')
