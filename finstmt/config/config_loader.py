import yaml
from dataclasses import asdict
from typing import List, Dict, Any
from finstmt.findata.item_config import ItemConfig
from finstmt.forecast.config import ForecastConfig

def load_yaml_config(filepath: str) -> List[Dict]:
    """
    Load raw configuration data from a YAML file
    
    :param filepath: Path to YAML config file
    :return: List of raw config dictionaries
    """
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)

def save_yaml_config(configs: List['StatementConfig'], filepath: str) -> None:
    """
    Save statement configurations to a YAML file
    
    :param configs: List of StatementConfig objects
    :param filepath: Path where to save the YAML file
    """
    config_data = {'statements': []}
    
    for stmt_config in configs:
        stmt_dict = {
            'key': stmt_config.key,
            'display_name': stmt_config.display_name,
            'items': []
        }
        
        for item in stmt_config.items_config_list:
            item_dict = {
                'key': item.key,
                'display_name': item.display_name,
                'extract_names': item.extract_names,
            }
            
            if item.expr_str:
                item_dict['expr_str'] = item.expr_str
            if not item.force_positive:
                item_dict['force_positive'] = item.force_positive
            if item.display_verbosity != 1:
                item_dict['display_verbosity'] = item.display_verbosity
            if item.forecast_config:
                item_dict['forecast'] = asdict(item.forecast_config)
                
            stmt_dict['items'].append(item_dict)
            
        config_data['statements'].append(stmt_dict)
        
    with open(filepath, 'w') as f:
        yaml.dump(config_data, f, sort_keys=False)