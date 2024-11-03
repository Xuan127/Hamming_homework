import time, os, logging, json
import google.generativeai as gemini
from deprecated.helper_structs import Discovery, NextStep
from deprecated.llm_functions import safety_settings
from pydantic import ValidationError
from DecisionTree import DecisionNode, DecisionEdge

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/llm_parsers.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_information(api_key: str, model_name: str, text: str, information_database: list[str]) -> str:
    """
    Parses and extracts key information from a given text using the specified generative model.
    
    Args:
        api_key (str): The API key for authenticating with the generative model.
        model_name (str): The name of the generative model to use.
        text (str): The text to be parsed.
        information_database (list[str]): A list of existing information to check for duplicates.
    
    Returns:
        str: The extracted key information or "DUPLICATE" if the information exists in the database.
    """
    logger.info("Starting parse_information")
    try:
        time.sleep(4)
        gemini.configure(api_key=api_key)
        logger.debug(f"Configured gemini with model {model_name}")
        
        system_instruction = f"""
        You are a natural language parser analyzing business AI agent statements.
        Your task is to extract and rephrase the key information from each statement based on its conversation state.
        For a statement, identify and return only the most relevant information.
        Keep the response concise and focused on the core meaning.

        Look at the information database and if the statement is found in the database, then return "DUPLICATE".
        Otherwise, return the most relevant information from the statement.
        Information database: {information_database}

        Examples:
        - "We typically accept major credit cards, checks, and cash." -> "The business accepts major credit cards, checks, and cash"
        - "We are located at 123 Main Street." -> "The business is located at 123 Main Street"
        - "Our business hours are from 9am to 5pm, Monday to Friday." -> "The business's business hours are from 9am to 5pm, Monday to Friday"
        """
        
        model = gemini.GenerativeModel(
            model_name,
            system_instruction=system_instruction,
        )
        logger.debug("GenerativeModel initialized")
        
        response = model.generate_content(
            f"The statement is {text}",
            safety_settings=safety_settings
        )
        logger.info("Content generated successfully")
        
        return response.text.strip()
    
    except Exception as e:
        logger.error(f"Error in parse_information: {e}", exc_info=True)
        raise

def parse_question(api_key: str, model_name: str, text: str, question_database: list[str]) -> str:
    """
    Parses and extracts key questions from a given text using the specified generative model.
    
    Args:
        api_key (str): The API key for authenticating with the generative model.
        model_name (str): The name of the generative model to use.
        text (str): The text to be parsed.
        question_database (list[str]): A list of existing questions to check for duplicates.
    
    Returns:
        str: The extracted question information or "DUPLICATE" if the question exists in the database.
    """
    logger.info("Starting parse_question")
    try:
        time.sleep(4)
        gemini.configure(api_key=api_key)
        logger.debug(f"Configured gemini with model {model_name}")
        
        system_instruction = f"""
        You are a natural language parser analyzing business AI agent statements.
        Your task is to extract and rephrase the key information from each statement based on its conversation state.
        For a statement, identify and return only the most relevant information.
        Keep the response concise and focused on the core meaning.

        Look at the question database and if the statement is found in the database, then return "DUPLICATE".
        Otherwise, return the most relevant information from the statement.
        Question database: {question_database}

        Examples:
        - "What are your working hours?" -> "The business will ask for the caller's working hours"
        - "Do you need any more information?" -> "The business will ask if the caller needs any more information"
        - "I'm sorry. I can help with that." -> "The business will offer to help with the caller's request"
        - "Are you an existing customer?" -> "The business will ask if the caller is an existing customer"
        """
        
        model = gemini.GenerativeModel(
            model_name,
            system_instruction=system_instruction,
        )
        logger.debug("GenerativeModel initialized")
        
        response = model.generate_content(
            f"The statement is {text}",
            safety_settings=safety_settings
        )
        logger.info("Content generated successfully")
        
        return response.text.strip()
    
    except Exception as e:
        logger.error(f"Error in parse_question: {e}", exc_info=True)
        raise

def parse_action(api_key: str, model_name: str, text: str, action_database: list[str]) -> str:
    """
    Parses and extracts action-related information from a given text using the specified generative model.
    
    Args:
        api_key (str): The API key for authenticating with the generative model.
        model_name (str): The name of the generative model to use.
        text (str): The text to be parsed.
        action_database (list[str]): A list of existing actions to check for duplicates.
    
    Returns:
        str: The extracted action information or "DUPLICATE" if the action exists in the database.
    """
    logger.info("Starting parse_action")
    try:
        time.sleep(4)
        gemini.configure(api_key=api_key)
        logger.debug(f"Configured gemini with model {model_name}")
        
        system_instruction = f"""
        You are a natural language parser analyzing business AI agent statements.
        Your task is to extract and rephrase the key information from each statement based on its conversation state.
        For a statement, identify and return only the most relevant information.
        Keep the response concise and focused on the core meaning.

        Look at the information database and if the statement is found in the database, then return "DUPLICATE".
        Otherwise, return the most relevant information from the statement.
        Action database: {action_database}

        Examples:
        - "I will schedule an appointment for you" -> "The business will schedule an appointment for the caller"
        - "I am going to call you back" -> "The business will call the caller back"
        """
        
        model = gemini.GenerativeModel(
            model_name,
            system_instruction=system_instruction,
        )
        logger.debug("GenerativeModel initialized")
        
        response = model.generate_content(
            f"The statement is {text}",
            safety_settings=safety_settings
        )
        logger.info("Content generated successfully")
        
        return response.text.strip()
    
    except Exception as e:
        logger.error(f"Error in parse_action: {e}", exc_info=True)
        raise

def parse_conversation(api_key: str, model_name: str, text: str, history: list[str] = []) -> str:
    """
    Analyzes a conversation transcript to determine the conversation state and summarizes it as a statement.
    
    Args:
        api_key (str): The API key for authenticating with the generative model.
        model_name (str): The name of the generative model to use.
        text (str): The current statement in the conversation to be analyzed.
        history (list[str], optional): A list of previous statements in the conversation. Defaults to an empty list.
    
    Returns:
        str: A string representing the conversation state and its summary.
    """
    logger.info("Starting parse_conversation")
    try:
        time.sleep(4)
        gemini.configure(api_key=api_key)
        logger.debug(f"Configured gemini with model {model_name}")
        
        system_instruction = f"""
        You are a natural language parser analyzing business AI agent statements.
        Analyze the conversation transcript and determine the conversation state and summarise it as a statement.
        The conversation state can be one of the following: QUESTION, ACTION, END, INFORMATION, CLARIFICATION, CONFIRMATION, ACTION_REQUEST, FILLER

        Look at the history statements and do not repeat the same state, ignore statements if they are in the history.
        For example, if the history contains a question on existing customers, then do not return a question on existing customers again.
        History: {history}

        State examples:
        - "Are you an existing customer?" -> "QUESTION"
        - "I will schedule an appointment for you" -> "ACTION"
        - "Thank you for calling. Goodbye!" -> "END"
        - "We typically accept major credit cards, checks, and cash." -> "INFORMATION"
        - "I'm not sure what you mean by that." -> "CLARIFICATION"
        - "Is that correct?" -> "CONFIRMATION"
        - "I can help with that." -> "ACTION_REQUEST"
        - "..." -> "FILLER"
        - "What do you need help with?" -> "FILLER"
        - "How else can I help you?" -> "FILLER"
        - "I can help you find another agent to assist you." -> "TRANSFER"
        - "We will have another agent call you back about your aircon servicing needs." -> "TRANSFER"

        Output example:
        - {Discovery(state="QUESTION", text="The agent asks if the caller is calling about an appointment.")}
        - {Discovery(state="ACTION", text="The agent will schedule an appointment for the caller")}
        - {Discovery(state="END", text="The agent thanks the caller for calling and says goodbye")}
        - {Discovery(state="INFORMATION", text="The agent provides information about the business's payment methods, which are major credit cards, checks, and cash")}
        - {Discovery(state="CLARIFICATION", text="The agent is asking for clarification on the caller's request")}
        - {Discovery(state="CONFIRMATION", text="The agent is confirming whether a statement is correct")}
        - {Discovery(state="ACTION_REQUEST", text="The agent is requesting more information from the caller before taking action")}
        - {Discovery(state="FILLER", text="...")}
        - {Discovery(state="TRANSFER", text="The agent will transfer the caller to another agent")}
        - {Discovery(state="TRANSFER", text="The agent asks the caller to find another agent to help them")}
        """
        
        model = gemini.GenerativeModel(
            model_name,
            system_instruction=system_instruction,
        )
        logger.debug("GenerativeModel initialized")
        
        generation_config = gemini.GenerationConfig(
            response_mime_type="application/json",
            response_schema=list[Discovery]
        )
        
        response = model.generate_content(
            f"The statement is {text}",
            safety_settings=safety_settings,
            generation_config=generation_config
        )
        logger.info("Content generated successfully")
        
        logger.debug(f"Response: {response.text}")
        return response.text.strip()
    
    except Exception as e:
        logger.error(f"Error in parse_conversation: {e}", exc_info=True)
        raise



GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = "gemini-1.5-flash"

if __name__ == "__main__":
    pass
