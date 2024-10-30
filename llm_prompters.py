import time, os
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
        6. Ask for the products and services that the business offers

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

def generate_next_prompt(api_key: str, model_name: str, business_description: str, question: str, response: str, history: list[str]) -> str:
    gemini.configure(api_key=api_key)

    model = gemini.GenerativeModel(model_name,
            system_instruction=f"""
            You are a prompt engineer specializing in creating system prompts for AI Voice Agents that will call and test businesses.
            The agent you are prompting for will be the caller, initiating conversations with the business to test their AI system.
            Given a business description, create a comprehensive system prompt that will help test all possible
            conversation paths and scenarios. The prompt should:

            1. Define the agent's role as a caller testing the business's AI system
            2. Specify a specific scenario to try, which is given in the form of a question and the response that the tester AI should give
            3. Include how to handle unexpected responses or system failures
            4. Specify a natural, conversational tone that mimics real customer calls
            5. You can give make up names and numbers, but nothing that is illegal or immoral

            Format the response as a clear, structured system prompt that can be used directly with an AI model.
            The tester agent's task is to specially explore the question that the business AI agent asked.
            Use the response to guide the tester agent's exploration, the tester agent should use the response to determine what to say next.

            The history of the conversation is given below: {history}
            The tester agent should use the history to determine what to say next.
            """)
    text = f"The business description is: {business_description}, the question is: {question}, the response is: {response}"
    response = model.generate_content(text, safety_settings=safety_settings)
    
    return response.text.strip()

if __name__ == "__main__":
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = 'gemini-1.5-flash'
    business_description = "A business that sells and services aircon units"
    question = 'The business will ask if the caller is an existing customer.'
    response = 'yes, I am an existing customer'
    print(generate_next_prompt(GEMINI_API_KEY, GEMINI_MODEL, business_description, question, response))
