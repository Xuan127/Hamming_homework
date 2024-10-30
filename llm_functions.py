import os, time
import google.generativeai as gemini
from helper_structs import Discovery, QuestionResponse

safety_settings={
        gemini.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: gemini.types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        gemini.types.HarmCategory.HARM_CATEGORY_HARASSMENT: gemini.types.HarmBlockThreshold.BLOCK_NONE,
        gemini.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: gemini.types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        gemini.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: gemini.types.HarmBlockThreshold.BLOCK_NONE,
        # gemini.types.HarmCategory.HARM_CATEGORY_UNSPECIFIED: gemini.types.HarmBlockThreshold.BLOCK_NONE
    }

def determine_state(api_key: str, model_name: str, sentence: str) -> str:
    """
    Determine the conversation state of a given text using Gemini AI.
    
    Args:
        sentence (str): The input text to analyze
        
    Returns:
        str: The determined state ('question', 'action', or 'end')
    """
    time.sleep(5)
    gemini.configure(api_key=api_key)

    model = gemini.GenerativeModel(
        model_name=model_name,
        system_instruction="""
        Analyze the given text and determine its conversation state.
        Possible states:
        - 'question': If the text is asking something or seeking information
        - 'action': If the text is a statement, command, or describing an action
        - 'end': If the text indicates a conclusion or termination
        - 'information': If the text is providing information or details
        - 'clarification': If the text is asking for or providing clarification
        - 'confirmation': If the text is confirming or seeking confirmation
        - 'action_request': If the text is specifically requesting an action be taken
        - 'filler': If the text is filler words or phrases that are not important to the conversation
        - 'transfer': If the text is indicating that the conversation should be transferred to another agent
        You should also return the text that is relevant to the conversation state.
        The text should be the most relevant part of the conversation that is relevant to the conversation state.
        You can return multiple sentences if they are all relevant to the conversation state.
        You can return multiple states if the text contains multiple conversation states.

        Examples of filler words or phrases:
        - "thank you", "okay", "that's great", "I understand", "I'm sorry", "I apologize", "hello", "hi", "how are you", "I see"
        - "this is XXX speaking"
        - "free to reach out anytime", "It was nice talking to you", "have a great day"
        - "How can I help you today?"

        Examples of action statements:
        - "I will schedule an appointment for you"
        - "I am going to call you back"
        - "Our agent will call you back"

        Examples of action requests:
        - "Can you send me a text with the address?"

        Examples of questions:
        - "What are your working hours?"
        - "Do you need any more information?"

        Examples of information:
        - "We typically accept major credit cards, checks, and cash."
        - "We are located at 123 Main Street."
        - "Our business hours are from 9am to 5pm, Monday to Friday."

        Examples of clarification:
        - "I'm not sure what you mean by that."
        - "Can you clarify what you meant by that?"

        Examples of confirmation:
        - "Is that correct?"
        - "Do you confirm that you want to proceed?"

        Examples of end:
        - "Thank you for calling. Goodbye!"
        - "I hope that helps. Goodbye!"

        Examples of transfer:
        - "I am transferring you to another agent."
        - "Please talk to the other department."
        """
    )

    response = model.generate_content(
        sentence,
        generation_config=gemini.GenerationConfig(
            response_mime_type="application/json",
            response_schema=list[Discovery]
        ),
        safety_settings=safety_settings
    )
    
    return response.text.strip().lower()

def identify_speaker(api_key: str, model_name: str, transcript: str) -> str:
    # Use Gemini to identify the business AI agent
    identify_speaker_prompt = """
    Analyze this conversation transcript and determine which speaker number represents the business representative/agent. 
    Look for clues like introducing the business name, answering customer queries professionally, or following business protocols.
    Return only the speaker number (e.g., '0' or '1').

    Transcript:
    """ + transcript

    gemini.configure(api_key=api_key)
    model = gemini.GenerativeModel(model_name)
    response = model.generate_content(identify_speaker_prompt, safety_settings=safety_settings)
    business_speaker = response.text.strip()

    return business_speaker

def generate_question_response(api_key: str, model_name: str, question: str, information_database: list[str] = []) -> str:
    time.sleep(5)
    gemini.configure(api_key=api_key)

    system_prompt = f"""
    You are given a question that a business AI agent asked a customer. Your job is to generate a list of possible responses in JSON format.
    The JSON should have two keys: "question" and "response".
    You can make use of existing in formation in {information_database} to generate a response.

    Examples:
    - If they ask if you are the customer, output a list of responses including "yes, I am the customer", "no, I am not the customer"
    - If they ask if you are calling about an appointment, output a list of responses including "yes, I am calling about an appointment", "no, I am not calling about an appointment"
    - If they ask if you are calling about a service, output a list of responses including "yes, I am calling about a service", "no, I am not calling about a service"

    Example output:
    [
        "question": "The agent asks if you are calling about an appointment.", "response": "yes, I am calling about an appointment",
        "question": "The agent asks if you are calling about an appointment.", "response": "no, I am not calling about an appointment","
    ]
    [
        question: "The agent asks if the caller is an existing customer", response: "yes, I am an existing customer",
        question: "The agent asks if the caller is an existing customer", response: "no, I am not an existing customer",
    ]
    [
        question: "The agent asks if the caller is calling about a service", response: "yes, I am calling about a service",
        question: "The agent asks if the caller is calling about a service", response: "no, I am not calling about a service",
    ]
    """
   
    model = gemini.GenerativeModel(model_name, system_instruction=system_prompt)
    response = model.generate_content("the question is: " + question, safety_settings=safety_settings,
        generation_config=gemini.GenerationConfig(
            response_mime_type="application/json",
            response_schema=list[QuestionResponse]))
    return response.text.strip()

def check_in_history(api_key: str, model_name: str, history: list[str], question: str) -> bool:
    time.sleep(5)
    gemini.configure(api_key=api_key)
    model = gemini.GenerativeModel(model_name,
        system_instruction="""
        You are checking if a question has been asked before in a conversation history.
        Return 'true' if the question is found in the history, 'false' otherwise.
        Compare the semantic meaning, not just exact matches.
        """)
    
    response = model.generate_content(
        f"Question: {question}\nHistory: {history}", 
        safety_settings=safety_settings,
        generation_config=gemini.GenerationConfig(
            response_mime_type="application/json",
            response_schema=bool))
    
    return response.text.strip().lower() == "true"

if __name__ == "__main__":
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    GEMINI_MODEL = "gemini-1.5-flash"

    # print(determine_state(GEMINI_API_KEY, GEMINI_MODEL, 
    #     """Thank you, Alex. I understand you're just gathering information. 
    #     We typically accept major credit cards, checks, and cash. 
    #     If you have any other questions or need further assistance, feel free to ask."""))

    print(generate_question_response(GEMINI_API_KEY, GEMINI_MODEL, 
        "The agent asks if you are calling about an appointment.", 
        ["The business is open from 9am to 5pm, Monday to Friday."]))
    print(generate_question_response(GEMINI_API_KEY, GEMINI_MODEL, 
        "The agent asks if you are calling about a service.", 
        ["The business is open from 9am to 5pm, Monday to Friday."]))