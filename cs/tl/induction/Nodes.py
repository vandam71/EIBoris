from __future__ import annotations
from typing import Union, TypedDict, Any


class DecisionNode:
    _id: int = 1

    def __init__(self, split_data: SplitNode, left: Union[DecisionNode, LeafNode], right: Union[DecisionNode, LeafNode], num_samples: int) -> None:
        self.split_data = split_data
        self.left = left
        self.right = right
        self.num_samples = num_samples
        self.id: int = DecisionNode._id
        DecisionNode._id += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "id": self.id,
            "split_data": self.split_data,
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
            "num_samples": self.num_samples,
        }


class LeafNode:
    _id: int = 1

    def __init__(self, label: int, num_samples: int) -> None:
        self.label = label
        self.num_samples = num_samples
        self.id: int = LeafNode._id
        LeafNode._id += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "id": self.id,
            "label": self.label,
            "num_samples": self.num_samples,
        }


class SplitNode(TypedDict):
    feature_index: int
    threshold: float
    metric_gain: float
