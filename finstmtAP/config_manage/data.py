from dataclasses import dataclass
from typing import Dict, List

from sympy import Idx, IndexedBase, symbols

from finstmt.config_manage.base import ConfigManagerBase
from finstmt.items.config import ItemConfig


@dataclass
class DataConfigManager(ConfigManagerBase):
    """
    Used to manage the config for an individual time period of an individual statement
    """

    configs: List[ItemConfig]

    def __post_init__(self):
        self.configs = list(self.configs)

    def __getitem__(self, item):
        return self.configs[item]

    def __iter__(self):
        yield from self.configs

    def get(self, item_key: str) -> ItemConfig:
        """
        Get entire configuration for item by key
        """
        return self.config_dict[item_key]

    def set(self, item_key: str, config: ItemConfig):
        """
        Set entire configuration for item by key
        """
        orig_config = self.get(item_key)
        config_idx = self.configs.index(orig_config)
        self.configs[config_idx] = config

    # TODO [#8]: Avoid unnecessary calculations in DataConfigManager
    #
    #  Make config_dict and sympy_namespace not properties, but recalculate any time config changes
    @property
    def config_dict(self) -> Dict[str, ItemConfig]:
        return {config.key: config for config in self.configs}

    @property
    def sympy_namespace(self) -> Dict[str, IndexedBase]:
        t = symbols("t", cls=Idx)
        ns_dict = {"t": t}
        for config in self.configs:
            expr = IndexedBase(config.key)
            ns_dict.update({config.key: expr})
            if config.forecast_config.pct_of is not None:
                key_pct_of_key = _key_pct_of_key(
                    config.key, config.forecast_config.pct_of
                )
                expr = IndexedBase(key_pct_of_key)
                ns_dict.update({key_pct_of_key: expr})
        return ns_dict

    @property
    def keys(self) -> List[str]:
        return list(self.config_dict.keys())

    @property
    def items(self) -> List[ItemConfig]:
        return self.configs


def _key_pct_of_key(base_key: str, pct_of_key: str) -> str:
    return f"{base_key}_pct_{pct_of_key}"
