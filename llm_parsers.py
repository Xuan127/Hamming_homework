import time
import google.generativeai as gemini
from helper_structs import Discovery
from llm_functions import safety_settings

def parse_information(api_key: str, model_name: str, text: str, information_database: list[str]) -> str:
    time.sleep(5)
    gemini.configure(api_key=api_key)

    model = gemini.GenerativeModel(model_name,
            system_instruction=f"""
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
            )
    response = model.generate_content(f"The statement is {text}", safety_settings=safety_settings)
    
    return response.text.strip()


def parse_question(api_key: str, model_name: str, text: str, question_database: list[str]) -> str:
    time.sleep(5)
    gemini.configure(api_key=api_key)

    model = gemini.GenerativeModel(model_name,
            system_instruction=f"""
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
            )
    response = model.generate_content(f"The statement is {text}", safety_settings=safety_settings)
    
    return response.text.strip()


def parse_action(api_key: str, model_name: str, text: str, action_database: list[str]) -> str:
    time.sleep(5)
    gemini.configure(api_key=api_key)

    model = gemini.GenerativeModel(model_name,
            system_instruction=f"""
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
            )
    response = model.generate_content(f"The statement is {text}", safety_settings=safety_settings)
    
    return response.text.strip()

def parse_conversation(api_key: str, model_name: str, text: str, history: list[str] = []) -> str:
    time.sleep(5)
    gemini.configure(api_key=api_key)

    model = gemini.GenerativeModel(model_name,
            system_instruction=f"""
            You are a natural language parser analyzing business AI agent statements.
            Analyze the conversation transcript and determine the conversation state and summarise it as a statement.
            The conversation state can be one of the following: QUESTION, ACTION, END, INFORMATION, CLARIFICATION, CONFIRMATION, ACTION_REQUEST, FILLER

            Look at the history statements and do not repeat the same state, ignore statements if they are in the history.
            For example, if the history contains a question on existing customers, then do not return a question on existing customers again.

            State examples:
            - "Hello, how can I help you today?" -> "QUESTION"
            - "I will schedule an appointment for you" -> "ACTION"
            - "Thank you for calling. Goodbye!" -> "END"
            - "We typically accept major credit cards, checks, and cash." -> "INFORMATION"
            - "I'm not sure what you mean by that." -> "CLARIFICATION"
            - "Is that correct?" -> "CONFIRMATION"
            - "I can help with that." -> "ACTION_REQUEST"
            - "..." -> "FILLER"

            Output example:
            - {Discovery(state="QUESTION", text="The agent asks if the caller is calling about an appointment.")}
            - {Discovery(state="ACTION", text="The agent will schedule an appointment for the caller")}
            - {Discovery(state="END", text="The agent thanks the caller for calling and says goodbye")}
            - {Discovery(state="INFORMATION", text="The agent provides information about the business's payment methods")}
            - {Discovery(state="CLARIFICATION", text="The agent is asking for clarification on the caller's request")}
            - {Discovery(state="CONFIRMATION", text="The agent is confirming whether a statement is correct")}
            - {Discovery(state="ACTION_REQUEST", text="The agent is requesting more information from the caller before taking action")}
            - {Discovery(state="FILLER", text="...")}
            """
            )
    response = model.generate_content(f"The statement is {text}", safety_settings=safety_settings,
        generation_config=gemini.GenerationConfig(
            response_mime_type="application/json",
            response_schema=Discovery))
    
    return response.text.strip()

