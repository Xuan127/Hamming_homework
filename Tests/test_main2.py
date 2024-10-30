import json
import os
from helper_functions import call_hamming_and_transcribe
from llm_functions import determine_state, identify_speaker, generate_question_response
from llm_parsers import parse_conversation

HAMMING_API_KEY = os.environ['HAMMING_API_KEY']
DEEPGRAM_API_KEY = os.environ['DEEPGRAM_API_KEY']
GEMINI_API_KEY = os.environ['GEMINI_API_KEY']
GEMINI_MODEL = "gemini-1.5-flash"
# number_to_call = os.environ['NUMBER_TO_CALL']
# business_description = "Air Conditioning and Plumbing"

# initial_prompt = generate_initial_prompt(GEMINI_API_KEY, GEMINI_MODEL, business_description)

# # Directly save the prompt to a file
# with open("generated_prompt.txt", "w") as f:
#     f.write(initial_prompt)

# call_hamming_and_transcribe(HAMMING_API_KEY, DEEPGRAM_API_KEY, number_to_call, initial_prompt)

# Read the transcription output
with open("transcription_output.txt", "r") as f:
    transcript = f.read()

business_speaker = identify_speaker(GEMINI_API_KEY, GEMINI_MODEL, transcript)

print(f"\nIdentified business AI agent as Speaker {business_speaker}")

history = [{"question": "The agent asks if the caller is an existing customer", "response": "yes, I am an existing customer"}]

response = parse_conversation(GEMINI_API_KEY, GEMINI_MODEL, f"the speaker is {business_speaker}\n{transcript}", history)
response = json.loads(response)
if response['state'] == 'question':
    print(generate_question_response(GEMINI_API_KEY, GEMINI_MODEL, response['text']))
elif response['state'] == 'action':
    print(f"Action: {response['text']}")
elif response['state'] == 'information':
    print(f"Information: {response['text']}")