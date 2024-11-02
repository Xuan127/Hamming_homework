import time, os, logging, json
from DecisionTree import DecisionNode, DecisionEdge, DecisionTree
from openai import OpenAI
import openai

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


def parse_tree(api_key: str, model_name: str, conversation: str, nodes: list[DecisionNode], edges: list[DecisionEdge], tool_choice) -> list[DecisionNode]:
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
        </context>

        <instructions>
        You are given a conversation between a business AI agent (callee) and a tester AI agent (caller).
        You are also given a list of nodes and edges that represent the current decision tree.
        Your task is to add new nodes and edges to the decision tree based on the given text if necessary.
        You should not add duplicate nodes to the decision tree, this includes the same and very similar nodes.
        You should only refer to the examples but not copying the content.
        </instructions>
        
        <node format>
        {"id": "1", "label": "Are you an existing customer?", "type": "question"}
        </node format>

        <edge format>
        {"source_id": "1", "target_id": "2", "condition": "yes"}
        </edge format>

        <decision tree format>
        nodes: [{"id": "1", "label": "Are you an existing customer?", "type": "question"},
        {"id": "2", "label": "transfer to Premium Concierge", "type": "action"}]
        edges: [{"source_id": "1", "target_id": "2", "condition": "yes"}]
        This will be a tree where the node 1 is the root node and the node 2 is the next node.
        </decision tree format>

        <example>
        Conversation:
        - [Speaker 0] Are you an existing customer?
        - [Speaker 1] Yes, I am a Gold Member.
        - [Speaker 0] I will transfer you to our Premium Concierge.

        Current decision tree:
        nodes: []
        edges: []

        In this example, you should:
        1. Add a new node with the label "existing customer?"
        2. Add a new node with the label "transfer to Premium Concierge"
        3. Connect the first node with the second node with an edge labeled "yes"
        The nodes should be connected in sequence to form a decision path.
        Output decision tree (it should strictly follow this format):
        nodes: [{"id": "1", "label": "existing customer?", "type": "question"}, 
        {"id": "2", "label": "transfer to Premium Concierge", "type": "action"}]
        edges: [{"source_id": "1", "target_id": "2", "condition": "yes"}]
        </example>
        
        <duplicate nodes example>
        - "existing customer?" and "existing customer with us?" are considered duplicate nodes.
        </duplicate nodes example>
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
            ],
            tools=[tool_choice]
        )
        return response.choices[0].message.tool_calls
    
    except Exception as e:
        print(e)

def get_nodes_and_edges(api_key: str, model_name: str, conversation: str, nodes: list[DecisionNode], edges: list[DecisionEdge]) -> tuple[list[DecisionNode], list[DecisionEdge]]:
    result = parse_tree(api_key, model_name, conversation, nodes, edges, draw_node_tool)
    if result:
        nodes = []
        for tool_call in result:
            node_data = json.loads(tool_call.function.arguments)
            nodes.append(node_data)
    result = parse_tree(api_key, model_name, conversation, nodes, edges, draw_edge_tool)
    if result:
        edges = []
        for tool_call in result:
            edge_data = json.loads(tool_call.function.arguments)
            edges.append(edge_data)
    return nodes, edges

def parse_nodes_and_edges(tree: DecisionTree, nodes, edges):
    for node in nodes:
        tree.add_node(node["id"], node["label"])
    for edge in edges:
        tree.add_edge(edge["source_id"], edge["target_id"], edge["condition"])
    return tree

if __name__ == "__main__":
    conversation = open("examples/transcription_1.txt", "r").read()
    nodes, edges = get_nodes_and_edges(os.environ.get("OPENAI_API_KEY"), "gpt-4o", conversation, [], [])
    print(f"Nodes: {nodes}")
    print(f"Edges: {edges}")
    tree = parse_nodes_and_edges(DecisionTree(), nodes, edges)
    tree.display()

