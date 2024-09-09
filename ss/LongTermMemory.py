from .MemoryNode import MemoryNode
from typing import List


class LongTermMemory:
    T_REMEMBER = 30  # Threshold for remembering
    T_RECOGNITION = 0.95  # Similarity threshold for recognition

    def __init__(self) -> None:
        """Initialize the LongTermMemory object."""
        self.memory: dict[str, MemoryNode] = {}  # Dictionary to store memory nodes

    def add_node(self, node: MemoryNode) -> None:
        """Add a node to the memory.
        Args:
            node (MemoryNode): The node to be added to the memory.
        """
        # Perform recognition process and update connected nodes' hotness
        for node_id in self.memory.keys():
            # Compute the similarity between the new node and existing nodes
            similarity = (node - self.memory[node_id]) * 230
            # print(similarity)
            # If the similarity meets the recognition threshold, add the connection
            if similarity >= self.T_RECOGNITION:
                node.connected_nodes[node_id] = similarity
        # Add the new node to the memory
        self.memory[node.id] = node

    def get_population(self, node: MemoryNode) -> List[MemoryNode]:
        """Retrieve the population of recognized nodes from memory based on the provided node.
        Args:
            node (MemoryNode): The node used for retrieval.
        Returns:
            List[MemoryNode]: The population of recognized nodes from memory.
        """
        # Retrieve the recognized nodes from memory based on the provided node
        population = [self.memory[key] for key in node.connected_nodes.keys() if self.memory[key].hot_index > self.T_REMEMBER]
        # Update the node core with the new hotness (increase the hotness of accessed items and decrease the hotness of non-accessed items)
        # When retrieving the nodes for the population, those are the nodes that need to be updated, everything else in memory needs to decay in hotness.
        # Identify the IDs of nodes that need to decay in hotness
        decay_hotness_ids = self.memory.keys() - set(node.connected_nodes.keys()) - {node.id}
        # Decrement the hotness of non-accessed nodes
        for key in decay_hotness_ids:
            self.memory[key].hot_index = max(self.memory[key].hot_index - 1, 0)
        # Increment the hotness of accessed nodes
        for key in node.connected_nodes.keys():
            self.memory[key].hot_index = min(self.memory[key].hot_index + 1, 100)
        return population

    def draw_ltm(self) -> None:
        import networkx as nx
        import matplotlib.pyplot as plt

        G = nx.Graph()
        for node in self.memory.values():
            G.add_node(node.id, hot_index=node.hot_index, color="lightgreen" if node.hot_index > self.T_REMEMBER else "gray")
        for node in self.memory.values():
            for node_id, weight in node.connected_nodes.items():
                G.add_edge(node.id, node_id, weight=round(weight, 3))
        # Draw the graph
        pos = nx.spring_layout(G)  # Choose a layout algorithm
        hot_index_labels = nx.get_node_attributes(G, "hot_index")  # Retrieve hot_index attribute values
        node_colors = nx.get_node_attributes(G, "color")
        nx.draw(G, pos, with_labels=True, labels=hot_index_labels, node_color=list(node_colors.values()), edge_color="gray", width=2, font_size=8, node_size=500)
        # Draw edge labels
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")
        # Show the graph
        plt.show()
