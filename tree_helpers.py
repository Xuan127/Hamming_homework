import os, datetime, logging, json
from DecisionTree import DecisionNode, DecisionEdge, DecisionTree
from openai import OpenAI
import openai
import streamlit as st

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        # logging.FileHandler(f"logs/tree_helpers_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        # logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

draw_node_tool = openai.pydantic_function_tool(DecisionNode)
draw_edge_tool = openai.pydantic_function_tool(DecisionEdge)

draw_node_tool["function"]["description"] = """To get a node in the decision tree."""
draw_node_tool["function"]["parameters"]["properties"]["id"]["description"] = """The id of the node. It is just a unique integer."""
draw_node_tool["function"]["parameters"]["properties"]["label"]["description"] = """The label of the node."""
draw_node_tool["function"]["parameters"]["$defs"]["DecisionNodeTypes"]["description"] = """The type of the node.
question: The node is a question asked by the callee agent.
action: The node is an action done by the callee agent.
inquiry: The node is an inquiry made by the caller.
"""

draw_edge_tool["function"]["description"] = """To get an edge in the decision tree."""
draw_edge_tool["function"]["parameters"]["properties"]["source_id"]["description"] = """The id of the source node."""
draw_edge_tool["function"]["parameters"]["properties"]["target_id"]["description"] = """The id of the target node."""
draw_edge_tool["function"]["parameters"]["properties"]["condition"]["description"] = """The condition of the edge."""

def parse_nodes_and_edges(api_key: str, model_name: str, conversation: str, nodes: list[DecisionNode], edges: list[DecisionEdge]) -> list[DecisionNode]:
    """
    Parses a given text into a predefined decision tree JSON structure using the specified generative model.

    Args:
        api_key (str): The API key for authenticating with the generative model.
        model_name (str): The name of the generative model to use.
        conversation (str): The current conversation to be analyzed.
        nodes (list[DecisionNode]): A list of nodes in the decision tree.
        edges (list[DecisionEdge]): A list of edges in the decision tree.

    Returns:
        list[DecisionNode]: A list of nodes in the decision tree.
    """
    try:
        logger.debug("Initializing OpenAI client.")
        client = OpenAI(api_key=api_key)

        system_instruction = """
        <context>
        We are trying to draw a decision tree for a business AI agent.
        Each node in the decision tree is either a question or an action.
        The decision describes the actions that the callee agent can take and the conditions that lead to different actions.
        Each inquiry is an inquiry made by the caller, they should not be sequential but should be parallel.
        Each inquiry should be from the same node.
        </context>

        <instructions>
        You are given a conversation between a business AI agent (callee) and a tester AI agent (caller).
        You are also given a list of nodes and edges that represent the current decision tree.
        Your task is to add new nodes and edges to the decision tree based on the given text if necessary.
        You should not add duplicate nodes to the decision tree, this includes the same and very similar nodes. If there are nodes with the same meaning, you should not add new ones.
        Each node should have a single action or question.
        Each node's label should be in third person or no person.
        The edges represents the decision path between the nodes.
        You should only refer to the examples but not copying the content.
        Do not have duplicate node ids.
        You can output zero nodes and edges if no new ones are needed.
        </instructions>
        
        <description>
        node: To get a node in the decision tree.
        'id' of node = The id of the node. It is just a unique integer.
        'label' of node = The label of the node.
        'type' of node = [question: The node is a question asked by the callee agent.
        action: The node is an action done by the callee agent.
        inquiry: The node is an inquiry made by the caller.
        ]
        edge: To get an edge in the decision tree.
        'source_id' of edge = The id of the source node.
        'target_id' of edge = The id of the target node.
        'condition' of edge = The condition of the edge, it can be a question asked by the caller or a condition that leads to the action.
        </description>

        <node format>
        {"id": "_", "label": "_", "type": "_"}
        </node format>

        <edge format>
        {"source_id": "_", "target_id": "_", "condition": "_"}
        </edge format>
        
        <duplicate nodes example>
        - "existing customer?" and "existing customer with us?" are considered duplicate nodes.
        </duplicate nodes example>

        <ignore these phrases>
        Ignore all filler words and phrases.
        Ignore all phrases that are not related to the business context.
        - "I am sorry"
        - "I am not sure"
        - "I do not know"
        - "I do not understand"
        - "how can i help you?"
        - "how can i assist you?"
        - "how can i help you today?"
        - "how can i assist you today?"
        - "Thank you"
        - "Have a great day!"
        - "Goodbye!"
        - "goodbye and thank you"
        - "Is there anything else I can help you with?"
        </ignore these phrases>
        """
        
        logger.debug("Creating chat completion request.")
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": system_instruction,
                },
                {
                    "role": "user",
                    "content": f"The conversation is {conversation}, current decision tree: [nodes: {nodes}, edges: {edges}]",
                },
            ]
        )
        logger.debug("Chat completion received successfully.")
        return response.choices[0].message.content
    
    except Exception as e:
        logger.error(f"Error in parse_nodes_and_edges: {e}", exc_info=True)
        return []

def get_nodes(api_key: str, model_name: str, text: str) -> list[DecisionNode]:
    """
    Extracts all nodes from the given text using the specified generative model.

    Args:
        api_key (str): The API key for authenticating with the generative model.
        model_name (str): The name of the generative model to use.
        text (str): The text to extract nodes from.

    Returns:
        list[DecisionNode]: A list of extracted nodes or None if no nodes are found.
    """
    try:
        logger.debug("Initializing OpenAI client for get_nodes.")
        client = OpenAI(api_key=api_key)

        logger.debug("Creating chat completion request for nodes extraction.")
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": """
                    Extract all the nodes from the given text.
                    The nodes should follow the format of {"id": _, "label": _, "type": _}
                    output none if there are no nodes.
                    """,
                },
                {
                    "role": "user",
                    "content": f"The text is {text}",
                },
            ],
            tools=[draw_node_tool],
            tool_choice="auto",
        )

        result = response.choices[0].message.tool_calls
        if result:
            nodes = []
            for tool_call in result:
                node_data = json.loads(tool_call.function.arguments)
                nodes.append(node_data)
            logger.debug(f"Extracted nodes: {nodes}")
            return nodes
        logger.debug("No nodes extracted.")
        return None

    except Exception as e:
        logger.error(f"Error in get_nodes: {e}", exc_info=True)
        return None

def get_edges(api_key: str, model_name: str, text: str) -> list[DecisionEdge]:
    """
    Extracts all edges from the given text using the specified generative model.

    Args:
        api_key (str): The API key for authenticating with the generative model.
        model_name (str): The name of the generative model to use.
        text (str): The text to extract edges from.

    Returns:
        list[DecisionEdge]: A list of extracted edges or None if no edges are found.
    """
    try:
        logger.debug("Initializing OpenAI client for get_edges.")
        client = OpenAI(api_key=api_key)

        logger.debug("Creating chat completion request for edges extraction.")
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": """
                    Extract all the edges from the given text.
                    The edges should follow the format of {"source_id": _, "target_id": _, "condition": _}
                    output none if there are no edges.
                    """,
                },
                {
                    "role": "user",
                    "content": f"The text is {text}",
                },
            ],
            tools=[draw_edge_tool],
            tool_choice="auto",
        )

        result = response.choices[0].message.tool_calls
        if result:
            edges = []
            for tool_call in result:
                edge_data = json.loads(tool_call.function.arguments)
                edges.append(edge_data)
            logger.debug(f"Extracted edges: {edges}")
            return edges
        logger.debug("No edges extracted.")
        return None

    except Exception as e:
        logger.error(f"Error in get_edges: {e}", exc_info=True)
        return None

def parse_tree(tree: DecisionTree, nodes: list[DecisionNode], edges: list[DecisionEdge]) -> DecisionTree:
    """
    Parses and updates the decision tree with new nodes and edges, ensuring no duplicates.

    Args:
        tree (DecisionTree): The decision tree to be updated.
        nodes (list[DecisionNode]): A list of nodes to add to the tree.
        edges (list[DecisionEdge]): A list of edges to add to the tree.

    Returns:
        DecisionTree: The updated decision tree.
    """
    try:
        logger.debug("Starting parse_tree process.")
        # Remove duplicate nodes by creating a set of node IDs that have been seen
        seen_node_ids = set()
        unique_nodes = []
        
        for node in nodes:
            node_id = node["id"]
            if node_id not in seen_node_ids:
                seen_node_ids.add(node_id)
                unique_nodes.append(node)
            else:
                logger.debug(f"Duplicate node found and skipped: {node}")

        nodes = unique_nodes
        for node in nodes:
            if node["type"] == "question":
                tree.add_decision_node(node["id"], node["label"])
                logger.debug(f"Added decision node: {node}")
            elif node["type"] == "action":
                tree.add_node(node["id"], node["label"])
                logger.debug(f"Added action node: {node}")
            elif node["type"] == "inquiry":
                tree.add_inquiry_node(node["id"], node["label"])
                logger.debug(f"Added inquiry node: {node}")
            else:
                logger.warning(f"Unknown node type encountered: {node}")

        for edge in edges:
            tree.add_edge(edge["source_id"], edge["target_id"], edge["condition"])
            logger.debug(f"Added edge: {edge}")

        logger.debug("parse_tree process completed successfully.")
        return tree

    except Exception as e:
        logger.error(f"Error in parse_tree: {e}", exc_info=True)
        return tree

if __name__ == "__main__":
    try:
        logger.info("Starting the Decision Tree application.")
        st.set_page_config(layout="wide")
        tree = DecisionTree()
        nodes = []
        edges = []

        conversation = open("examples/transcription_1.txt", "r").read()
        logger.info("Parsed conversation from transcription_1.txt")

        text = parse_nodes_and_edges(os.environ.get("OPENAI_API_KEY"), "o1-preview", conversation, nodes, edges)
        with open("output1.txt", "w") as f:
            f.write(str(text))
        logger.info("Wrote parse_nodes_and_edges output to output1.txt")

        new_nodes = get_nodes(os.environ.get("OPENAI_API_KEY"), "gpt-4o", text)
        new_edges = get_edges(os.environ.get("OPENAI_API_KEY"), "gpt-4o", text)
        logger.info(f"Extracted new nodes: {new_nodes}")
        logger.info(f"Extracted new edges: {new_edges}")

        if new_nodes is None:
            new_nodes = []
            logger.debug("No new nodes to add.")

        if new_edges is None:
            new_edges = []
            logger.debug("No new edges to add.")

        nodes.extend(new_nodes)
        edges.extend(new_edges)
        tree = parse_tree(tree, new_nodes, new_edges)
        tree.display()
        logger.info("Decision tree displayed successfully.")

    except Exception as e:
        logger.critical(f"Unhandled exception in main: {e}", exc_info=True)
