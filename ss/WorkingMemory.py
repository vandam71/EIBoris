import numpy as np
from typing import List
from .MemoryNode import MemoryNode


class WorkingMemory:
    def __init__(self, size: int) -> None:
        """Initialize the WorkingMemory object.
        Args:
            size (int): The maximum size of the working memory.
        """
        self.memory: list[MemoryNode] = []  # List to store memory nodes
        self.size: int = size  # Maximum size of the working memory

    def add_node(self, node: MemoryNode) -> None:
        """Updates the memory with a new node.
        If the node is already in the memory, it is moved to the most recent location.
        If the memory is full, the oldest node is replaced with the new node.
        Args:
            node (Node): The node to update the memory with.
        """
        if node in self.memory:
            # If the node is already in the memory, remove it to re-add at a more recent location
            self.memory.remove(node)
        elif len(self.memory) >= self.size:
            self.memory.pop(0)  # If the memory is full, remove the oldest node (at index 0)
        self.memory.append(node)  # Add the new node to the memory

    def get_influence(self) -> List[float]:
        """Calculate the influence based on the final decisions, hot index, and similarity factors.
        Returns:
            List[float]: The normalized influence values.
        """
        # First iteration where the memory only has one node
        if len(self.memory) == 1:
            return [0.5, 0.5, 0.5]
        total_influence = []  # Initialize empty influence list
        for node in self.memory[:-1]:
            # Use hot index to balance the final decision
            base_influence = np.array(node.data.final_decision) * (node.hot_index / 100)
            if node.id in self.memory[-1].connected_nodes.keys():
                # Use similarity to further balance the influence
                base_influence *= 1 - self.memory[-1].connected_nodes[node.id]
            total_influence.append(base_influence)  # Add this node's influence to the total influence
        # Calculate the total influence, normalize it and return
        return (np.sum(total_influence, axis=0) / np.max(np.sum(total_influence, axis=0))).tolist()

    def __repr__(self) -> str:
        return str(self.memory)
