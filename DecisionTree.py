import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
from pydantic import BaseModel
from enum import Enum
from typing import Optional

class DecisionNodeTypes(Enum):
    QUESTION = "question"
    ACTION = "action"
    INQUIRY = "inquiry"

class DecisionNode(BaseModel):
    id: str
    type: DecisionNodeTypes
    label: str

class DecisionEdge(BaseModel):
    source_id: str
    target_id: str
    condition: Optional[str]

class DecisionTree:
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.nodes_kwargs = {
            "font": {"color": 'white'}
        }
        self.edges_kwargs = {

        }
        self.config = Config(width=1750,
            height=750,
            directed=True, 
            physics=False, 
            hierarchical={"enabled": True, "direction": "UD"},
            improvedLayout=True,
            nodeSpacing=5000,
            levelSeparation=500
            )

    def add_node(self, id, label):
        self.nodes.append(Node(id=id, label=label, size=25, shape="dot", color="red", **self.nodes_kwargs))

    def add_inquiry_node(self, id, label):
        self.nodes.append(Node(id=id, label=label, size=25, shape="dot", color="green", **self.nodes_kwargs))

    def add_decision_node(self, id, label):
        self.nodes.append(Node(id=id, label=label, size=25, shape="diamond", color="blue", **self.nodes_kwargs))

    def add_edge(self, source, target, label):
        self.edges.append(Edge(source=source, target=target, label=label, **self.edges_kwargs))

    def get_nodes_as_dict(self):
        return [{"id": node.id, "label": node.label} for node in self.nodes]

    def get_edges_as_dict(self):
        return [{"source": edge.source, "target": edge.target, "label": edge.label} for edge in self.edges]

    def wrap_label(self, label, max_length=20):
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
        return wrapped_label.strip()
    
    def display(self):
        temp_nodes = self.nodes.copy()
        for node in temp_nodes:
            node.label = self.wrap_label(node.label)
        temp_edges = self.edges.copy()
        for edge in temp_edges:
            edge.label = self.wrap_label(edge.label)
        agraph(nodes=temp_nodes, edges=temp_edges, config=self.config)

if __name__ == "__main__":
    tree = DecisionTree()
    tree.add_node("Spiderman", "Peter Parker")
    tree.add_node("Captain_Marvel", "Carol Danvers")
    tree.add_edge("Captain_Marvel", "Spiderman", "friend_of")
    tree.add_decision_node("Do you know Spiderman?", "Do you know Spiderman?")
    tree.add_edge("Do you know Spiderman?", "Spiderman", "yes")
    tree.add_edge("Do you know Spiderman?", "Captain_Marvel", "no")
    tree.display()

