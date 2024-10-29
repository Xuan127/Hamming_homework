import os
import google.generativeai as gemini
from helper_structs import Discovery

def determine_state(api_key: str, model_name: str, sentence: str) -> Discovery:
    """
    Determine the conversation state of a given text using Gemini AI.
    
    Args:
        sentence (str): The input text to analyze
        
    Returns:
        str: The determined state ('question', 'action', or 'end')
    """
    gemini.configure(api_key=api_key)

    model = gemini.GenerativeModel(
        model_name=model_name,
        system_instruction="""
        Analyze the given text and determine its conversation state.
        Return only one of these states: 'question', 'action', 'end', 'information', 'clarification', 'confirmation', or 'action_request'.
        - 'question': If the text is asking something or seeking information
        - 'action': If the text is a statement, command, or describing an action
        - 'end': If the text indicates a conclusion or termination
        - 'information': If the text is providing information or details
        - 'clarification': If the text is asking for or providing clarification
        - 'confirmation': If the text is confirming or seeking confirmation
        - 'action_request': If the text is specifically requesting an action be taken
        """
    )

    response = model.generate_content(
        sentence,
        generation_config=gemini.GenerationConfig(
            response_mime_type="application/json",
            response_schema=list[Discovery]
        )
    )
    
    return response.text.strip().lower()

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


if __name__ == "__main__":
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    GEMINI_MODEL = "gemini-1.5-pro"

    print(determine_state(GEMINI_API_KEY, GEMINI_MODEL, "Write a story about a backpack."))

    # Example usage
    business_desc = """
    Local auto repair shop specializing in routine maintenance, repairs, and diagnostics.
    Services include oil changes, brake service, tire rotation, and general repairs.
    Needs to handle appointment scheduling, service inquiries, and basic cost estimates.
    """
    
    system_prompt = generate_initial_prompt(GEMINI_API_KEY, GEMINI_MODEL, business_desc)
    print("\nGenerated System Prompt:")
    print(system_prompt)

