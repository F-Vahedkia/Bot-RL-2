# f03_features/feature_B_graph.py
# reviewed at 1405/04/28
"""
در این فایل موارد زیر باید اضافه بشوند:
    - dependencies
    - edges
    - topology
    - execution order
    - groups

"""
# ============================================================
# Imports
# ============================================================
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple
from collections import defaultdict
from f10_utils.parser import parse_spec

# ============================================================
# Node
# ============================================================
@dataclass(frozen=True)
class FeatureNode:
    raw: str                # مثال:  'sma(column="close",period=100)@M4'  همان (رشته-اندیکاتور) خام است.
    name: str               # مثال:  'sma'
    timeframe: str | None   # مثال:  'M10'

# ============================================================
# Feature Graph
# ============================================================
class FeatureGraph:
    """
    FeatureGraph
    مسئول:
    - خواندن config
    - تبدیل لیست indicators به graph
    - استخراج dependency + grouping  --> هنوز انجام نشده است
    """
    # -------------------------------------------------------- 1
    def __init__(self, specs: List[str]):
        self.specs = specs

        self.nodes: List[FeatureNode] = []
        self.by_tf: Dict[str, List[FeatureNode]] = defaultdict(list)
        self.by_name: Dict[str, List[FeatureNode]] = defaultdict(list)

        self._build()

    # -------------------------------------------------------- 2
    def _build(self):

        for s in self.specs:
            try:
                ps = parse_spec(s)
            except Exception:
                continue
        
            node = FeatureNode(
                raw=ps.raw,      # .replace("'", '"'),
                name=ps.name,
                timeframe=ps.timeframe
            )

            self.nodes.append(node)

            tf = ps.timeframe or "GLOBAL"

            self.by_tf[tf].append(node)
            self.by_name[ps.name].append(node)

    # -------------------------------------------------------- 3
    def get_by_timeframe(self, tf: str) -> List[FeatureNode]:
        return self.by_tf.get(tf, [])

    # -------------------------------------------------------- 4
    def get_by_name(self, name: str) -> List[FeatureNode]:
        return self.by_name.get(name, [])

    # -------------------------------------------------------- 5
    def all_nodes(self) -> List[FeatureNode]:
        return self.nodes

# ============================================================ END
