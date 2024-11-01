import time, os, logging, json
from groq import Groq
import google.generativeai as gemini
from DecisionTree import DecisionNode, DecisionEdge

safety_settings={
    gemini.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: gemini.types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    gemini.types.HarmCategory.HARM_CATEGORY_HARASSMENT: gemini.types.HarmBlockThreshold.BLOCK_NONE,
    gemini.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: gemini.types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    gemini.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: gemini.types.HarmBlockThreshold.BLOCK_NONE,
    # gemini.types.HarmCategory.HARM_CATEGORY_UNSPECIFIED: gemini.types.HarmBlockThreshold.BLOCK_NONE
    }

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('logs/parsers.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def parse_nodes(api_key: str, model_name: str, conversation: str, nodes: list[DecisionNode], edges: list[DecisionEdge]) -> list[DecisionNode]:
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
    logger.info("Starting parse_nodes")
    try:
        gemini.configure(api_key=api_key)
        logger.debug(f"Configured gemini with model {model_name}")
        
        system_instruction = """
        You are a decision tree parser that draws a decision tree from conversations between a business AI agent (callee) and a tester AI agent(caller).

        You are given a list of nodes and edges that represent the decision tree. The output should be this format too, follow it strictly.
        Do not add any other text to the output including comments.
        Example of a node: {"id": "1", "label": "Are you an existing customer?", "is_question": True}
        Example of an edge: {"source": "1", "target": "2", "label": "yes"}
        This will be a tree where the node 1 is the root node and the node 2 is the next node.

        Your task is to add new nodes to the decision tree based on the given text if necessary.
        The decision tree should only be the questions and actions that the callee AI asks or does.
        Example:
        - [Speaker 0] Are you an existing customer?
        - [Speaker 1] Yes, I am a Gold Member.
        - [Speaker 0] I will transfer you to our Premium Concierge.
        In this example, you should:
        1. Add a new node with the label "existing customer?" and connect it to the previous node
        2. Add a new node with the label "transfer to Premium Concierge" and connect it to the first node with an edge labeled "yes"
        The nodes should be connected in sequence to form a decision path.
        Output nodes (it should strictly follow this format): [{"id": "1", "label": "existing customer?", "is_question": True}, 
        {"id": "2", "label": "transfer to Premium Concierge", "is_question": False}]

        You should not add duplicate nodes to the decision tree, this includes the same and very similar nodes.
        Examples of duplicate nodes:
        - "existing customer?" and "existing customer with us?"

        The new node should be a question node if the text contains a question, otherwise it should be an action node.

        The new node should be added to the existing nodes and the nodes should be returned.

        DO NOT REPEAT THE OUTPUT FORMAT.
        """
        
        model = gemini.GenerativeModel(
            model_name,
            system_instruction=system_instruction,
        )
        logger.debug("GenerativeModel initialized")
        
        generation_config = gemini.GenerationConfig(
            response_mime_type="application/json",
            response_schema=list[DecisionNode]
        )
        
        response = model.generate_content(
            f"The conversation is {conversation}, nodes: {nodes}, edges: {edges}",
            safety_settings=safety_settings,
            generation_config=generation_config
        )
        logger.info("Content generated successfully")
        
        logger.debug(f"Response: {response.text}")
        json_nodes = json.loads(response.text.strip())
        return [DecisionNode(**node) for node in json_nodes]
    
    except Exception as e:
        logger.error(f"Error in parse_nodes: {e}", exc_info=True)

def parse_edges(api_key: str, model_name: str, conversation: str, nodes: list[DecisionNode], edges: list[DecisionEdge]) -> list[DecisionEdge]:
    """
    Parses a given text into a predefined decision tree JSON structure using the specified generative model.
    
    Args:
        api_key (str): The API key for authenticating with the generative model.
        model_name (str): The name of the generative model to use.
        conversation (str): The current conversation to be analyzed.
        nodes (list[DecisionNode]): A list of nodes in the decision tree.
        edges (list[DecisionEdge]): A list of edges in the decision tree.
    
    Returns:
        list[DecisionEdge]: A list of edges in the decision tree.
    """
    logger.info("Starting parse_edges") 
    try:
        gemini.configure(api_key=api_key)
        logger.debug(f"Configured gemini with model {model_name}")
        
        system_instruction = """
        You are a decision tree parser that draws a decision tree from conversations between a business AI agent (callee) and a tester AI agent(caller).

        You are given a list of nodes and edges that represent the decision tree. The output should be this format too, follow it strictly.
        Do not add any other text to the output including comments.
        Example of a node: {"id": "1", "label": "Are you an existing customer?", "is_question": True}
        Example of an edge: {"source": "1", "target": "2", "label": "yes"}
        This will be a tree where the node 1 is the root node and the node 2 is the next node.

        Your task is to add new edges to the decision tree based on the given text if necessary.
        Example:
        - [Speaker 0] Are you an existing customer?
        - [Speaker 1] Yes, I am a Gold Member.
        - [Speaker 0] I will transfer you to our Premium Concierge.
        In this example, you should:
        1. Add an edge from "existing customer?" to "transfer to Premium Concierge" with the label "yes"
        Output edges (it should strictly follow this format): [{"source": "1", "target": "2", "label": "yes"}]

        You should not add duplicate edges to the decision tree.
        Examples of duplicate edges:
        - "yes" from "existing customer?" to "transfer to Premium Concierge" and "yes" from "existing customer with us?" to "transfer to Premium Concierge"

        The edges should connect nodes in sequence to form a decision path.
        The edge label should reflect the response that leads from one node to another.

        The new edges should be added to the existing edges and the edges should be returned.

        DO NOT REPEAT THE OUTPUT FORMAT.
        """
        
        model = gemini.GenerativeModel(
            model_name,
            system_instruction=system_instruction,
        )
        logger.debug("GenerativeModel initialized")
        
        generation_config = gemini.GenerationConfig(
            response_mime_type="application/json",
            response_schema=list[DecisionEdge]
        )
        
        response = model.generate_content(
            f"The conversation is {conversation}, nodes: {nodes}, edges: {edges}",
            safety_settings=safety_settings,
            generation_config=generation_config
        )
        logger.info("Content generated successfully")
        
        logger.debug(f"Response: {response.text}")
        json_edges = json.loads(response.text.strip())
        return [DecisionEdge(**edge) for edge in json_edges]
    
    except Exception as e:
        logger.error(f"Error in parse_edges: {e}", exc_info=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = "gemini-1.5-pro"

def test_parse():
    nodes = parse_nodes(GEMINI_API_KEY, GEMINI_MODEL_NAME, open('examples/transcription_1.txt', 'r').read(), [], [])
    edges = parse_edges(GEMINI_API_KEY, GEMINI_MODEL_NAME, open('examples/transcription_1.txt', 'r').read(), nodes, []) 
    print(nodes)
    print(edges)

if __name__ == "__main__":
    test_parse()
