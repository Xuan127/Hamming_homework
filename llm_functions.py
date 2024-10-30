import os, time
import google.generativeai as gemini
from helper_structs import Discovery

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


if __name__ == "__main__":
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    GEMINI_MODEL = "gemini-1.5-flash"

    print(determine_state(GEMINI_API_KEY, GEMINI_MODEL, 
        """Thank you, Alex. I understand you're just gathering information. 
        We typically accept major credit cards, checks, and cash. 
        If you have any other questions or need further assistance, feel free to ask."""))
