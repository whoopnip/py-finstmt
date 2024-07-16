from typing import Any, Dict, List, Set

from sympy import Eq, Expr, IndexedBase, sympify

from finstmt.exc import InvalidBalanceConfigException, NotACalculatedItemException
from finstmt.items.config import ItemConfig


class ConfigManagerBase:
    def get(self, item_key: str) -> ItemConfig:
        """
        For internal use, get the config as well as the key of the financial statement type it belongs to
        """
        raise NotImplementedError

    def set(self, item_key: str, config: ItemConfig) -> None:
        """
        Set entire configuration for item by key. Needs to handle setting the value in each individual
        data config manager
        """
        raise NotImplementedError

    @property
    def sympy_namespace(self) -> Dict[str, IndexedBase]:
        """
        The sympy namespace containing all the variable definitions
        """
        raise NotImplementedError

    @property
    def items(self) -> List[ItemConfig]:
        """
        All the configuration items
        """
        raise NotImplementedError

    def get_value(self, item_key: str, config_key: str) -> Any:
        """
        Get a particular configuration for a particular item
        """
        config = self.get(item_key)
        return getattr(config, config_key)

    def set_value(self, item_key: str, config_key: str, value: Any):
        """
        Set a particular configuration for a particular item
        """
        orig_config = self.get(item_key)
        setattr(orig_config, config_key, value)
        self.set(item_key, orig_config)

    def expr_for(self, item_key: str) -> Expr:
        config = self.get(item_key)
        if config.expr_str is None:
            raise NotACalculatedItemException(item_key)
        expr = sympify(config.expr_str, locals=self.sympy_namespace)
        return expr

    def eqs_involving(self, item_key: str, include_self_eq: bool = False) -> List[Eq]:
        ns = self.sympy_namespace
        item_sym = ns[item_key]
        t = ns["t"]
        item_t = item_sym[t]
        eqs: List[Eq] = []
        for config in self.items:
            if config.expr_str is None:
                continue
            rhs = self.expr_for(config.key)
            if item_t in rhs.free_symbols:
                this_item_sym = ns[config.key]
                this_item_sym_t = this_item_sym[t]
                eq = Eq(this_item_sym_t, rhs)
                eqs.append(eq)

        if include_self_eq:
            try:
                rhs = self.expr_for(item_key)
                eq = Eq(item_t, rhs)
                eqs.append(eq)
            except NotACalculatedItemException:
                pass

        return eqs

    def _calculated_item_determinant_keys(
        self, item_key: str, for_forecast: bool = True
    ) -> List[str]:
        determinant_keys: List[str] = []
        to_process_keys: List[str] = [item_key]
        i = -1
        while len(to_process_keys) > 0:
            i += 1
            process_key = to_process_keys.pop()
            if i != 0:
                # Add to determinants if not original key
                determinant_keys.append(process_key)
            eqs_involving_key = self.eqs_involving(process_key, include_self_eq=True)
            for eq in eqs_involving_key:
                all_eq_keys = self._expr_to_keys(eq.rhs - eq.lhs)
                new_keys = [
                    key
                    for key in all_eq_keys
                    if key != process_key
                    and key not in to_process_keys + determinant_keys
                ]
                if for_forecast:
                    accepted_keys: List[str] = []
                    for key in new_keys:
                        if self.get(key).forecast_config.make_forecast:
                            # As forecast is being made for this item, it is not calculated,
                            # and so shouldn't be processed. It should still be added to determinants
                            determinant_keys.append(key)
                            continue
                        accepted_keys.append(key)
                    new_keys = accepted_keys
                to_process_keys.extend(new_keys)
        return determinant_keys

    def item_determinant_keys(
        self, item_key: str, include_pct_of: bool = True, for_forecast: bool = True
    ) -> List[str]:
        determinant_keys = self._calculated_item_determinant_keys(
            item_key, for_forecast=for_forecast
        )
        if include_pct_of:
            for item in self.items:
                # TODO [$5fed05c64df698000808428a]: multiple passes through determinants may be necessary for complicated pct_of structures
                if (
                    item.key in determinant_keys
                    and item.forecast_config.pct_of is not None
                ):
                    pct_conf = self.get(item.forecast_config.pct_of)
                    if pct_conf.expr_str is None:
                        determinant_keys.append(item.forecast_config.pct_of)
                    else:
                        determinant_keys.extend(
                            self._calculated_item_determinant_keys(pct_conf.key)
                        )
        return list(set(determinant_keys))

    @property
    def balance_groups(self) -> List[Set[str]]:
        balance_sets: List[Set[str]] = []
        for item in self.items:
            if item.forecast_config.balance_with is not None:
                # Skip items already tracked
                item_tracked = False
                for bl in balance_sets:
                    # Balance item in a group which was already already found, skip
                    if item.key in bl:
                        item_tracked = True
                        break
                if item_tracked:
                    continue

                # Not already tracked, must be a new balance group
                balance_group: Set[str] = {item.key, item.forecast_config.balance_with}
                balance_with_conf = self.get(item.forecast_config.balance_with)
                if balance_with_conf.forecast_config.balance_with != item.key:
                    # This is part of a balance chain, e.g. a balanced with b, b balanced with c, c balanced with a
                    changed = True
                    while changed:
                        num_balanced = len(balance_group)
                        balance_with_key = (
                            balance_with_conf.forecast_config.balance_with
                        )
                        if balance_with_key is None:
                            raise InvalidBalanceConfigException(
                                f"{balance_with_conf.key} is part of balance group {balance_group} but in its "
                                f"forecast_config it has None for balance_with. Set balance_with for "
                                f"{balance_with_conf.key} to be another key in the balance group"
                            )
                        balance_group.add(balance_with_key)
                        new_num_balanced = len(balance_group)
                        changed = num_balanced != new_num_balanced
                        balance_with_conf = self.get(balance_with_key)

                balance_sets.append(balance_group)
        return balance_sets

    def _expr_to_keys(self, expr: Expr) -> List[str]:
        ns = self.sympy_namespace
        t = ns["t"]
        syms = expr.free_symbols
        keys: List[str] = []
        for key, key_expr in ns.items():
            if key == "t":
                continue
            key_expr_t = key_expr[t]
            if key_expr_t in syms:
                keys.append(key)
        return keys

    def eq_subs_dict(
        self, values_dict: Dict[str, float], t_offset: int = 0
    ) -> Dict[IndexedBase, float]:
        out_dict = {}
        t = self.sympy_namespace["t"]
        for item_key, item_symbol in self.sympy_namespace.items():
            if item_key in values_dict:
                indexed_symbol = item_symbol.__getitem__(
                    t + t_offset
                )  # eg cash[t] or cash[t-1]
                out_dict[indexed_symbol] = values_dict[item_key]
        return out_dict
