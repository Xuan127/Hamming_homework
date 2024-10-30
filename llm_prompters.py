import time
import google.generativeai as gemini
from llm_functions import safety_settings

def generate_initial_prompt(api_key: str, model_name: str, business_description: str) -> str:
    """
    Generate an initial system prompt for testing AI Voice Agents based on business description.
    
    Args:
        api_key (str): The Gemini API key
        model_name (str): The Gemini model to use
        business_description (str): Brief description of the business and its services
        
    Returns:
        str: Generated system prompt for the test agent
    """
    gemini.configure(api_key=api_key)

    model = gemini.GenerativeModel(
        model_name=model_name,
        system_instruction="""
        You are a prompt engineer specializing in creating system prompts for AI Voice Agents that will call and test businesses.
        The agent you are prompting for will be the caller, initiating conversations with the business to test their AI system.
        Given a business description, create a comprehensive system prompt that will help test all possible
        conversation paths and scenarios. The prompt should:

        1. Define the agent's role as a caller testing the business's AI system
        2. Specify various test scenarios to try (scheduling, inquiries, edge cases)
        3. Include how to handle unexpected responses or system failures
        4. Specify a natural, conversational tone that mimics real customer calls
        5. You can give make up names and numbers, but nothing that is illegal or immoral

        Format the response as a clear, structured system prompt that can be used directly with an AI model.
        """
    )

    prompt_template = f"""
    Based on the following business description, create a detailed system prompt for an AI Voice Agent that will test 
    the business's AI agent's conversational capabilities:
    {business_description}
    
    The prompt should help discover all possible conversation paths and edge cases during testing. Don't talk to me at all.
    """

    response = model.generate_content(prompt_template)
    return response.text.strip()

def generate_next_prompt_question(api_key: str, model_name: str, text: str, question_database: list[str], business_description: str) -> str:
    time.sleep(5)
    gemini.configure(api_key=api_key)

    model = gemini.GenerativeModel(model_name,
            system_instruction=f"""
            You are a prompt engineer specializing in creating system prompts for AI Voice Agents that will call and test businesses.
            The agent you are prompting for will be the caller, initiating conversations with the business to test their AI system.
            Given a business description, create a comprehensive system prompt that will help test all possible
            conversation paths and scenarios. The prompt should:

            1. Define the agent's role as a caller testing the business's AI system
            2. Specify various test scenarios to try
            3. Include how to handle unexpected responses or system failures
            4. Specify a natural, conversational tone that mimics real customer calls
            5. You can give make up names and numbers, but nothing that is illegal or immoral

            Format the response as a clear, structured system prompt that can be used directly with an AI model.
            Your task is to specially explore the question or action request that the business AI agent asked.
            If it provides multiple options, explore one of the options. You can also explore options that are not provided.
            You can also explore the information database to provide more information to the business AI agent.

            Examples:
            - If they ask if you are the customer, output a list of responses including "yes, I am the customer", "no, I am not the customer"
            - If they ask if you are calling about an appointment, output a list of responses including "yes, I am calling about an appointment", "no, I am not calling about an appointment", "I am calling about a service", "I am calling about a product"
            - If they ask if you are calling about a service, output a list of responses including "yes, I am calling about a service", "no, I am not calling about a service", "I am calling about a product", "I am calling about an appointment"

            Question database: {question_database}
            """)
    text = f"The business description is: {business_description}, the question is: {text}"
    response = model.generate_content(text, safety_settings=safety_settings, 
        generation_config=gemini.GenerationConfig(
            response_mime_type="application/json",
            response_schema=list[str]))
    
    return response.text.strip()