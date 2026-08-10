# f03_features/feature_B_resolver.py

from __future__ import annotations
from typing import Dict, List
import pandas as pd
from ...f10_utils.parser import parse_spec
from ..feature_B_registry_1 import get_indicator


class FeatureResolver:
    """
    DSL → required columns ONLY
    (strict schema mapping layer)
    """

    def __init__(self, available_columns: List[str]):
        self.columns = set(available_columns)

    # ---------------------------------------------------------
    def resolve(self, spec_str: str) -> Dict[str, List[str]]:

        parsed = parse_spec(spec_str)

        # registry (single source of truth)
        spec = get_indicator(parsed.name, "train")

        if spec is None:
            return {"inputs": [], "missing": []}

        required = spec.required_cols

        inputs = []
        missing = []

        tf = (parsed.timeframe or "").upper()

        for col in required:

            # DataHandler convention mapping
            mapped_col = (
                col if col.startswith(tf + "_")
                else f"{tf}_{col}"
            )

            if mapped_col in self.columns:
                inputs.append(mapped_col)
            else:
                missing.append(mapped_col)

        return {
            "inputs": inputs,
            "missing": missing
        }


def resolve_feature_columns(df: pd.DataFrame, spec_str: str):
    return FeatureResolver(list(df.columns)).resolve(spec_str)




""" CONTRACT TEXT FOR feature_B_resolver.py:

A) PUBLIC API MAP

**Class**

* `FeatureResolver`

  * `__init__(available_columns: List[str])`
  * `resolve(spec_str: str) -> Dict[str, List[str]]`

**Functions**

* `resolve_feature_columns(df: pandas.DataFrame, spec_str: str)`

  * Wrapper around `FeatureResolver.resolve`

**Entry Points**

* Module-level function:

  * `resolve_feature_columns(df, spec_str)`

---

B) DEPENDENCY CONTRACTS

**Standard Library**

* `typing: Dict, List`
* `__future__.annotations`

**External Libraries**

* `pandas as pd`

  * Used for `DataFrame` input in `resolve_feature_columns`

**Internal Modules**

* `f03_features.indicators.parser.parse_spec`

  * Parses DSL string into structured object with fields:

    * `name`
    * `timeframe` (optional)

* `f03_features.feature_B_registry.get_indicator`

  * Resolves indicator specification in `"train"` mode
  * Returns `IndicatorSpec | None`

**Internal Dependency Contracts**

* `parse_spec(spec_str)` MUST return object with `.name` and `.timeframe`
* `get_indicator(name, "train")` MUST support batch/train mode only
* `IndicatorSpec.required_cols` defines required input schema

---

C) DATA CONTRACT

**Input Types**

* `FeatureResolver.__init__`

  * `available_columns: List[str]`
  * Represents flattened dataset column names (already prefixed or raw)

* `resolve(spec_str: str)`

  * `spec_str: str`
  * DSL feature specification string (parsed via `parse_spec`)

* `resolve_feature_columns(df: pd.DataFrame, spec_str: str)`

  * `df.columns: Index[str]`

---

**Output Types**

* `resolve() -> Dict[str, List[str]]`

  * Schema:

    * `"inputs": List[str]`
    * `"missing": List[str]`

* `resolve_feature_columns() -> Dict[str, List[str]]`

  * Same schema as `resolve()`

---

**Internal Data Schema**

* `parsed` (from `parse_spec`)

  * `.name: str`
  * `.timeframe: Optional[str]`

* `IndicatorSpec` (from registry)

  * `.required_cols: List[str]`

* Column transformation rule:

  * `mapped_col = "{TF}_{col}"` if not already prefixed

---

**Data Constraints**

* If indicator not found:

  * return `{"inputs": [], "missing": []}`

* Column resolution rule:

  * strict prefix mapping using uppercase timeframe
  * no fallback resolution beyond prefix check

* Output separation:

  * `inputs ∩ available_columns`
  * `missing = required_cols not in available_columns (after mapping)`

---

**Error Contracts**

* No explicit exceptions raised
* Missing indicator → silent empty contract result
* Missing parsed fields → UNKNOWN CONTRACT FIELD

"""

