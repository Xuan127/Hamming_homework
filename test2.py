import time, os, logging, json
from DecisionTree import DecisionNode, DecisionEdge, DecisionTree
from openai import OpenAI
import openai
import streamlit as st
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


def parse_nodes_and_edges(api_key: str, model_name: str, conversation: str, nodes: list[DecisionNode], edges: list[DecisionEdge]):
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
        client = OpenAI(
            api_key=api_key,
        )
        
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
        inquiry: The node is an inquiry made by the caller.]
        edge: To get an edge in the decision tree.
        'source_id' of edge = The id of the source node.
        'target_id' of edge = The id of the target node.
        'condition' of edge = The condition of the edge.
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
        return response.choices[0].message.content
    
    except Exception as e:
        print(e)

def get_nodes(api_key: str, model_name: str, text: str) -> list[DecisionNode]:
    client = OpenAI(
        api_key=api_key,
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": """
                Extract all the nodes from the given text.
                The nodes should follow the format of {"id": _, "label": _, "type": _}
                output none if there are no new nodes.
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
        return nodes
    return None

def get_edges(api_key: str, model_name: str, text: str) -> list[DecisionEdge]:
    client = OpenAI(
        api_key=api_key,
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": """
                Extract all the edges from the given text.
                The edges should follow the format of {"source_id": _, "target_id": _, "condition": _}
                output none if there are no new edges.
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
        return edges
    return None
def parse_tree(tree: DecisionTree, nodes, edges):
    for node in nodes:
        if node["type"] == "question":
            tree.add_decision_node(node["id"], node["label"])
        elif node["type"] == "action":
            tree.add_node(node["id"], node["label"])
        elif node["type"] == "inquiry":
            tree.add_inquiry_node(node["id"], node["label"])
    for edge in edges:
        tree.add_edge(edge["source_id"], edge["target_id"], edge["condition"])
    return tree

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    tree = DecisionTree()
    nodes = []
    edges = []
    
    conversation = open("examples/transcription_1.txt", "r").read()
    text = parse_nodes_and_edges(os.environ.get("OPENAI_API_KEY"), "o1-preview", conversation, nodes, edges)
    with open("output1.txt", "w") as f:
        f.write(str(text))
    new_nodes = get_nodes(os.environ.get("OPENAI_API_KEY"), "gpt-4o", text)
    new_edges = get_edges(os.environ.get("OPENAI_API_KEY"), "gpt-4o", text)
    print(new_nodes)
    print(new_edges)
    if new_nodes == None: new_nodes = []
    if new_edges == None: new_edges = []
    nodes = nodes + new_nodes
    edges = edges + new_edges
    tree = parse_tree(tree, new_nodes, new_edges)
    tree.display()

    conversation = open("examples/transcription_2.txt", "r").read()
    text = parse_nodes_and_edges(os.environ.get("OPENAI_API_KEY"), "o1-preview", conversation, nodes, edges)
    with open("output2.txt", "w") as f:
        f.write(str(text))
    new_nodes = get_nodes(os.environ.get("OPENAI_API_KEY"), "gpt-4o", text)
    new_edges = get_edges(os.environ.get("OPENAI_API_KEY"), "gpt-4o", text)
    print(new_nodes)
    print(new_edges)
    if new_nodes == None: new_nodes = []
    if new_edges == None: new_edges = []
    nodes = nodes + new_nodes
    edges = edges + new_edges
    tree = parse_tree(tree, new_nodes, new_edges)
    tree.display()

    conversation = open("examples/transcription_3.txt", "r").read()
    text = parse_nodes_and_edges(os.environ.get("OPENAI_API_KEY"), "o1-preview", conversation, nodes, edges)
    with open("output3.txt", "w") as f:
        f.write(str(text))
    new_nodes = get_nodes(os.environ.get("OPENAI_API_KEY"), "gpt-4o", text)
    new_edges = get_edges(os.environ.get("OPENAI_API_KEY"), "gpt-4o", text)
    print(new_nodes)
    print(new_edges)
    if new_nodes == None: new_nodes = []
    if new_edges == None: new_edges = []
    nodes = nodes + new_nodes
    edges = edges + new_edges
    tree = parse_tree(tree, new_nodes, new_edges)
    tree.display()

    conversation = open("examples/transcription_4.txt", "r").read()
    text = parse_nodes_and_edges(os.environ.get("OPENAI_API_KEY"), "o1-preview", conversation, nodes, edges)
    with open("output4.txt", "w") as f:
        f.write(str(text))
    new_nodes = get_nodes(os.environ.get("OPENAI_API_KEY"), "gpt-4o", text)
    new_edges = get_edges(os.environ.get("OPENAI_API_KEY"), "gpt-4o", text)
    print(new_nodes)
    print(new_edges)
    if new_nodes == None: new_nodes = []
    if new_edges == None: new_edges = []
    nodes = nodes + new_nodes
    edges = edges + new_edges
    tree = parse_tree(tree, new_nodes, new_edges)
    tree.display()
