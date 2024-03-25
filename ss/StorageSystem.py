from utils import Singleton, add_logging
from .MemoryNode import MemoryNode
from .WorkingMemory import WorkingMemory
from .LongTermMemory import LongTermMemory


class StorageSystem(metaclass=Singleton):
    """
    This class works as a connection between the working and long-term memory.
    It calls all operations that need to be performed during the encoding, recognition, retrieval, and decision-making process.
    It is implemented as a singleton class, allowing it to be accessed by multiple places in the system without creating a new instance.
    """

    def __init__(self, wm_size: int) -> None:
        """Initialize the StorageSystem object.
        Args:
            wm_size (int): The size of the working memory.
        """
        self.working_memory = WorkingMemory(wm_size)  # Create a working memory object
        self.long_term_memory = LongTermMemory()  # Create a long-term memory object
        self._most_recent_node: MemoryNode = None  # Store the most recent memory node

    @property
    def most_recent_node(self) -> MemoryNode:
        """Getter method for the most_recent_node property.
        Returns:
            MemoryNode: The most recent memory node.
        """
        return self._most_recent_node

    @most_recent_node.setter
    @add_logging
    def most_recent_node(self, node: MemoryNode):
        """Setter method for the most_recent_node property.
        Args:
            node (MemoryNode): The memory node to set as the most recent node.
        """
        self._most_recent_node = node
        # Create/add new node to the memories
        # Connect the new node with nodes in the LTM that are similar and above the threshold (first stage of recognition)
        self.long_term_memory.add_node(self._most_recent_node)
        # Populate the working memory with similar nodes (n nodes, the most similar, depending on the working memory size)
        population = self.long_term_memory.get_population(self._most_recent_node)
        # Add the similar nodes to the working memory (maximum number of nodes to be added is one less than the working memory size)
        for node in reversed(population[: self.working_memory.size - 1]):
            # By reversing this list, the most similar node is added last and closest to the inferred node
            self.working_memory.add_node(node)
        # Add the most recent node to the beginning of the working memory
        self.working_memory.add_node(self._most_recent_node)
