from __future__ import annotations
import uuid
import torch
from typing import Optional
from dataclasses import dataclass
from utils import SortedDict


class MemoryNode:
    @dataclass
    class Data:
        low_level_features: torch.Tensor
        high_level_features: torch.Tensor
        probabilities: Optional[list] = None
        classification: Optional[list] = None
        final_decision: Optional[list] = None

        def __repr__(self) -> str:
            return f"Data(low_level_features={self.low_level_features.shape}, high_level_features={self.high_level_features.shape}, probabilities={self.probabilities}, classification={self.classification})"

    def __init__(self, data: set):
        """Initialize a MemoryNode instance.
        Args:
            data (set): A set containing the required data attributes for the MemoryNode.
        """
        self.connected_nodes = SortedDict()  # Dictionary to store connected nodes
        self.data = MemoryNode.Data(*data)  # Initialize the data attribute
        self.id = uuid.uuid4().hex  # Generate a unique ID for the node
        self.hot_index: int = 50  # Set the hot index to 50 (default value)

    def __repr__(self) -> str:
        """Return a string representation of the MemoryNode instance."""
        return f"Node(id={self.id}, connected_nodes={len(self.connected_nodes)}, hot_index={self.hot_index}, data={self.data})"

    def __sub__(self, other: MemoryNode) -> float:
        """Compute the similarity between two MemoryNode instances.
        Args:
            other (MemoryNode): The other MemoryNode instance to compare similarity with.
        Returns:
            float: The similarity score between the two MemoryNode instances.
        """
        low_level_similarity: torch.Tensor = 1 / (1 + torch.norm(self.data.low_level_features - other.data.low_level_features))
        high_level_similarity: torch.Tensor = 1 / (1 + torch.norm(self.data.high_level_features - other.data.high_level_features))
        return ((low_level_similarity + high_level_similarity).mean()).item()
