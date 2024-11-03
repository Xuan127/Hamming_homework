import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
from pydantic import BaseModel
from enum import Enum
from typing import Optional, List
import logging
import datetime

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        # logging.FileHandler(f'logs/decision_tree_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        # logging.StreamHandler()
    ]
)
logger = logging.getLogger()

class DecisionNodeTypes(Enum):
    """Enumeration of possible decision node types."""
    QUESTION = "question"
    ACTION = "action"
    INQUIRY = "inquiry"

class DecisionNode(BaseModel):
    """Model representing a node in the decision tree."""
    id: str
    type: DecisionNodeTypes
    label: str

class DecisionEdge(BaseModel):
    """Model representing an edge between nodes in the decision tree."""
    source_id: str
    target_id: str
    condition: Optional[str]

class DecisionTree:
    """
    A class to represent and manage a decision tree structure.

    Attributes:
        nodes (List[Node]): List of nodes in the tree.
        edges (List[Edge]): List of edges connecting the nodes.
        nodes_kwargs (dict): Additional keyword arguments for node styling.
        edges_kwargs (dict): Additional keyword arguments for edge styling.
        config (Config): Configuration for the agraph visualization.
    """

    def __init__(self):
        """
        Initializes the DecisionTree with empty nodes and edges lists.
        Sets up default styling and configuration.
        """
        self.nodes: List[Node] = []
        self.edges: List[Edge] = []
        self.nodes_kwargs = {
            "font": {"color": 'white'}
        }
        self.edges_kwargs = {}
        self.config = Config(
            width=1750,
            height=750,
            directed=True, 
            physics=False, 
            hierarchical={"enabled": True, "direction": "UD"},
            improvedLayout=True,
            nodeSpacing=5000,
            levelSeparation=500
        )
        logger.info("Initialized DecisionTree.")

    def add_node(self, id: str, label: str):
        """
        Adds a standard node to the decision tree.

        Args:
            id (str): Unique identifier for the node.
            label (str): Display label for the node.
        """
        try:
            node = Node(id=id, label=label, size=25, shape="dot", color="red", **self.nodes_kwargs)
            self.nodes.append(node)
            logger.info(f"Added node: {id} with label: {label}")
        except Exception as e:
            logger.error(f"Error adding node {id}: {e}")

    def add_inquiry_node(self, id: str, label: str):
        """
        Adds an inquiry node to the decision tree.

        Args:
            id (str): Unique identifier for the inquiry node.
            label (str): Display label for the inquiry node.
        """
        try:
            node = Node(id=id, label=label, size=25, shape="dot", color="green", **self.nodes_kwargs)
            self.nodes.append(node)
            logger.info(f"Added inquiry node: {id} with label: {label}")
        except Exception as e:
            logger.error(f"Error adding inquiry node {id}: {e}")

    def add_decision_node(self, id: str, label: str):
        """
        Adds a decision node to the decision tree.

        Args:
            id (str): Unique identifier for the decision node.
            label (str): Display label for the decision node.
        """
        try:
            node = Node(id=id, label=label, size=25, shape="diamond", color="blue", **self.nodes_kwargs)
            self.nodes.append(node)
            logger.info(f"Added decision node: {id} with label: {label}")
        except Exception as e:
            logger.error(f"Error adding decision node {id}: {e}")

    def add_edge(self, source: str, target: str, label: str):
        """
        Adds an edge between two nodes in the decision tree.

        Args:
            source (str): ID of the source node.
            target (str): ID of the target node.
            label (str): Label for the edge condition.
        """
        try:
            edge = Edge(source=source, target=target, label=label, type="CURVE_SMOOTH", **self.edges_kwargs)
            self.edges.append(edge)
            logger.info(f"Added edge from {source} to {target} with label: {label}")
        except Exception as e:
            logger.error(f"Error adding edge from {source} to {target}: {e}")

    def get_nodes_as_dict(self) -> List[dict]:
        """
        Retrieves all nodes as a list of dictionaries.

        Returns:
            List[dict]: List containing node information.
        """
        try:
            nodes_dict = [{"id": node.id, "label": node.label} for node in self.nodes]
            logger.debug(f"Nodes as dict: {nodes_dict}")
            return nodes_dict
        except Exception as e:
            logger.error(f"Error retrieving nodes as dict: {e}")
            return []

    def get_edges_as_dict(self) -> List[dict]:
        """
        Retrieves all edges as a list of dictionaries.

        Returns:
            List[dict]: List containing edge information.
        """
        try:
            edges_dict = [{"source": edge.source, "target": edge.target, "label": edge.label} for edge in self.edges]
            logger.debug(f"Edges as dict: {edges_dict}")
            return edges_dict
        except Exception as e:
            logger.error(f"Error retrieving edges as dict: {e}")
            return []

    def wrap_label(self, label: str, max_length: int = 20) -> str:
        """
        Wraps a label string to a specified maximum length per line.

        Args:
            label (str): The label string to wrap.
            max_length (int): Maximum number of characters per line.

        Returns:
            str: The wrapped label string.
        """
        try:
            if label is None:
                return ""
            label = label.replace('\n', ' ')
            words = label.split(' ')
            wrapped_label = ''
            current_length = 0
            for word in words:
                if current_length + len(word) + 1 > max_length:
                    wrapped_label += '\n' + word + ' '
                    current_length = len(word) + 1
                else:
                    wrapped_label += word + ' '
                    current_length += len(word) + 1
            wrapped_label = wrapped_label.strip()
            logger.debug(f"Wrapped label: Original: '{label}' | Wrapped: '{wrapped_label}'")
            return wrapped_label
        except Exception as e:
            logger.error(f"Error wrapping label '{label}': {e}")
            return label

    def display(self):
        """
        Displays the decision tree using Streamlit's agraph component.
        """
        try:
            temp_nodes = self.nodes.copy()
            for node in temp_nodes:
                node.label = self.wrap_label(node.label)
            temp_edges = self.edges.copy()
            for edge in temp_edges:
                if edge.label:
                    edge.label = self.wrap_label(edge.label)
            agraph(nodes=temp_nodes, edges=temp_edges, config=self.config)
            logger.info("Displayed the decision tree.")
        except Exception as e:
            logger.error(f"Error displaying the decision tree: {e}")

if __name__ == "__main__":
    try:
        tree = DecisionTree()
        tree.add_node("Spiderman", "Peter Parker")
        tree.add_node("Captain_Marvel", "Carol Danvers")
        tree.add_edge("Captain_Marvel", "Spiderman", "friend_of")
        tree.add_decision_node("Do you know Spiderman?", "Do you know Spiderman?")
        tree.add_edge("Do you know Spiderman?", "Spiderman", "yes")
        tree.add_edge("Do you know Spiderman?", "Captain_Marvel", "no")
        tree.display()
    except Exception as main_e:
        logger.critical(f"An unexpected error occurred: {main_e}")

