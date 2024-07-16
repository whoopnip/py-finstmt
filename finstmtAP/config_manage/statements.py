import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence, Tuple, Union

from sympy import IndexedBase

from finstmt.config_manage.base import ConfigManagerBase
from finstmt.config_manage.statement import StatementConfigManager
from finstmt.exc import NoSuchItemException
from finstmt.items.config import ItemConfig
from finstmt.logger import logger


@dataclass
class StatementsConfigManager(ConfigManagerBase):
    """
    Main configuration interface. Handles all of the configuration for a set of financial statements.
    """

    config_managers: Dict[str, StatementConfigManager]

    def get(self, item_key: str) -> ItemConfig:
        """
        Get entire configuration for item by key
        """
        config, _ = self._get(item_key)
        return config

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

    def _get(self, item_key: str) -> Tuple[ItemConfig, str]:
        """
        For internal use, get the config as well as the key of the financial statement type it belongs to
        """
        for fin_statement_type, manager in self.config_managers.items():
            try:
                return manager.get(item_key), fin_statement_type
            except NoSuchItemException:
                continue
        raise NoSuchItemException(item_key)

    def set(self, item_key: str, config: ItemConfig):
        """
        Set entire configuration for item by key
        """
        orig_config, fin_statement_key = self._get(item_key)
        self.config_managers[fin_statement_key].set(item_key, config)

    def update(self, item_key: str, config_keys: Union[str, Sequence[str]], value: Any):
        """
        Update configuration for item by item key and nested config keys

        :param item_key:
        :param config_keys:
        :param value:
        :return:
        """
        if isinstance(config_keys, str):
            config_keys = [config_keys]

        orig_config, fin_statement_key = self._get(item_key)
        nested_config = orig_config
        for i, config_key in enumerate(config_keys):
            if i == len(config_keys) - 1:
                # Last iteration, now set value
                setattr(nested_config, config_key, value)
                logger.debug(
                    f"Set {config_key} for {item_key} on {type(nested_config)} to {value}"
                )
            else:
                # Not last iteration, need to get nested config
                nested_config = getattr(nested_config, config_key)
        self.set(item_key, orig_config)

    def update_all(self, config_keys: Union[str, Sequence[str]], value: Any):
        """
        Update configuration for all items by nested config keys

        :param config_keys:
        :param value:
        :return:
        """
        for item_key in self.keys:
            self.update(item_key, config_keys, value)

    @property
    def sympy_namespace(self) -> Dict[str, IndexedBase]:
        ns_dict = {}
        for cfg_mgr in self.config_managers.values():
            ns_dict.update(cfg_mgr.sympy_namespace)
        return ns_dict

    @property
    def keys(self) -> List[str]:
        all_keys = set()
        for manager in self.config_managers.values():
            all_keys.update(manager.keys)
        return list(all_keys)

    @property
    def items(self) -> List[ItemConfig]:
        all_items = []
        for manager in self.config_managers.values():
            # Get unique maintaining order within a statement
            for item in manager.items:
                if item in all_items:
                    continue
                all_items.append(item)
        return all_items

    def dict(self) -> dict:
        item_data: Dict[str, dict] = {}
        for item in self.items:
            item_data[item.key] = asdict(item)
        return item_data

    def json(self, **kwargs) -> str:
        return json.dumps(self.dict(), **kwargs)
