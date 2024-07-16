from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
from sympy import IndexedBase

from finstmt.config_manage.base import ConfigManagerBase
from finstmt.config_manage.data import DataConfigManager
from finstmt.exc import NoSuchItemException
from finstmt.items.config import ItemConfig
from finstmt.logger import logger


@dataclass
class StatementConfigManager(ConfigManagerBase):
    """
    Used for entire single financial statement, e.g. income statement or balance sheet, with multiple dates in the
    statement.
    """

    config_managers: Dict[pd.Timestamp, DataConfigManager]

    def get(self, item_key: str) -> ItemConfig:
        """
        For internal use, get the config as well as the key of the financial statement type it belongs to
        """
        for date, manager in self.config_managers.items():
            try:
                conf = manager.get(item_key)
                logger.debug(f"Got config for {item_key} from {date}")
                return conf
            except KeyError:
                continue
        raise NoSuchItemException(item_key)

    def __getattr__(self, item_key: str) -> ItemConfig:
        """
        Get the config for a given key
        """
        # When copying the object, config_managers can be undefined and so would cause a
        # recursive loop without this check
        if item_key == "config_managers":
            # Trigger the default Python behavior
            return object.__getattribute__(self, item_key)

        try:
            return self.get(item_key)
        except NoSuchItemException:
            raise AttributeError(item_key)

    def __dir__(self) -> List[str]:
        return self.keys

    def set(self, item_key: str, config: ItemConfig) -> None:
        """
        Set entire configuration for item by key. Needs to handle setting the value in each individual
        data config manager
        """
        for date, manager in self.config_managers.items():
            manager.set(item_key, config)
            logger.debug(f"Set config for {item_key} for {date}")

    @property
    def sympy_namespace(self) -> Dict[str, IndexedBase]:
        for manager in self.config_managers.values():
            # All should be identical, so first is enough
            return manager.sympy_namespace
        raise ValueError("no managers, could not get sympy_namespace")

    @property
    def keys(self) -> List[str]:
        all_keys = set()
        for manager in self.config_managers.values():
            all_keys.update(manager.keys)
        return list(all_keys)

    @property
    def items(self) -> List[ItemConfig]:
        for manager in self.config_managers.values():
            return manager.configs
        return []
        # TODO [#39]: rethink low-level config structure
        #
        # We have a config for each item for each date and this method just gets the ones for the first date
