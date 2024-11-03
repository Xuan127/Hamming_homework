import os, logging
import google.generativeai as gemini
from deprecated.llm_functions import safety_settings

logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/llm_prompters.log"),
            logging.StreamHandler()
        ]
)

logger = logging.getLogger(__name__)
def generate_initial_prompt(api_key: str, model_name: str, business_description: str) -> str:
    """
    Generate an initial system prompt for testing AI Voice Agents based on business description.
    
    Args:
        api_key (str): The Gemini API key
        model_name (str): The Gemini model to use
        business_description (str): Brief description of the business and its services
        
    Returns:
        str: Generated system prompt for the test agent
        
    Raises:
        ValueError: If api_key, model_name or business_description is empty/None
        Exception: For any other errors during prompt generation
    """
    # Input validation
    if not api_key or not model_name or not business_description:
        logger.error("Missing required parameters")
        raise ValueError("api_key, model_name and business_description are required")

    try:
        logger.info(f"Generating initial prompt for model {model_name}")
        gemini.configure(api_key=api_key)

        model = gemini.GenerativeModel(
            model_name=model_name,
            system_instruction="""
            You are a prompt engineer specializing in creating system prompts for AI Voice Agents that will call and test businesses.
            The agent you are prompting for will be the caller, initiating conversations with the business to test their AI system.
            Given a business description, create a comprehensive system prompt that will help test all possible
            conversation paths and scenarios. The prompt should:

            1. Define the agent's role as a caller testing the business's AI system
            2. Specify various test scenarios to try (scheduling, inquiries, edge cases, or anything else)
            3. Include how to handle unexpected responses or system failures
            4. Specify a natural, conversational tone that mimics real customer calls
            5. You can give make up names and numbers, but nothing that is illegal or immoral
            6. Ask for the products and services that the business offers
            7. The prompt should help discover all possible conversation paths and edge cases during testing. Don't talk to me at all.
            8. The caller agent should not disclose that it is a tester agent, it should not say that it is testing the business's AI system.
            9. The prompt should be in markdown format.

            Format the response as a clear, structured system prompt that can be used directly with an AI model.
            """
        )

        prompt_template = f"""
        Based on the following business description, create a detailed system prompt for an AI Voice Agent that will test 
        the business's AI agent's conversational capabilities:
        {business_description}
        
        The prompt should help discover all possible conversation paths and edge cases during testing. Don't talk to me at all.
        """

        logger.debug(f"Sending prompt template to model: {prompt_template}")
        response = model.generate_content(prompt_template)
        
        generated_prompt = response.text.strip()
        logger.info("Successfully generated initial prompt")
        logger.debug(f"Generated prompt: {generated_prompt}")
        return generated_prompt

    except Exception as e:
        logger.error(f"Error generating initial prompt: {str(e)}")

def generate_next_prompt(api_key: str, model_name: str, business_description: str, question: str, response: str, history: list[str]=[]) -> str:
    # Input validation
    if not api_key or not model_name or not business_description or not question or not response:
        logger.error("Missing required parameters")
        raise ValueError("api_key, model_name, business_description, question and response are required")

    try:
        logger.info(f"Generating next prompt for model {model_name}")
        gemini.configure(api_key=api_key)

        model = gemini.GenerativeModel(model_name,
                system_instruction=f"""
                You are a prompt engineer specializing in creating system prompts for AI Voice Agents that will call and test businesses.
                The agent you are prompting for will be the caller, initiating conversations with the business to test their AI system.
                Given a business description, create a comprehensive system prompt that will help test all possible
                conversation paths and scenarios. The prompt should:

                1. Define the agent's role as a caller testing the business's AI system
                2. Specify various test scenarios to try (based on the question and response)
                3. Include how to handle unexpected responses or system failures
                4. Specify a natural, conversational tone that mimics real customer calls
                5. You can give make up names and numbers, but nothing that is illegal or immoral
                6. Ask for the products and services that the business offers
                7. The prompt should help discover all possible conversation paths and edge cases during testing. Don't talk to me at all.
                8. The caller agent should not disclose that it is a tester agent, it should not say that it is testing the business's AI system.
                9. The prompt should be in markdown format.

                Format the response as a clear, structured system prompt that can be used directly with an AI model.
                The tester agent's task is to specially explore the question that the business AI agent asked.
                Use the response to guide the tester agent's exploration, the tester agent should use the response to determine what to say next.

                The history of the conversation is given below: {history}
                The tester agent should use the history to determine what to say next.
                """)

        text = f"""The business description is: {business_description}, 
            what the agent is supposed to discover this call is when the business AI agent asks: {question}, 
            the response from the tester agent is: {response}
            """
        logger.debug(f"Sending prompt to model: {text}")
        
        response = model.generate_content(text, safety_settings=safety_settings)
        generated_prompt = response.text.strip()
        
        logger.info("Successfully generated next prompt")
        logger.debug(f"Generated prompt: {generated_prompt}")
        return generated_prompt

    except Exception as e:
        logger.error(f"Error generating next prompt: {str(e)}")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = 'gemini-1.5-pro'

def test_generate_initial_prompt():
    business_description = "A business that sells and services aircon units"
    prompt = generate_initial_prompt(GEMINI_API_KEY, GEMINI_MODEL, business_description)
    print(prompt)

def test_generate_next_prompt_no_history():
    business_description = "A business that sells and services aircon units"
    question = "Are you an existing customer?"
    response = "Yes, I am an existing customer"
    history = []
    prompt = generate_next_prompt(GEMINI_API_KEY, GEMINI_MODEL, business_description, question, response, history)
    print(prompt)

def test_generate_next_prompt_with_history():
    business_description = "A business that sells and services aircon units"
    question = "Is this an emergency call?"
    response = "No, it is not an emergency call"
    history = [{'question': "Are you an existing customer?", 'response': "Yes, I am an existing customer"}]
    prompt = generate_next_prompt(GEMINI_API_KEY, GEMINI_MODEL, business_description, question, response, history)
    print(prompt)

if __name__ == "__main__":
    test_generate_next_prompt_no_history()
