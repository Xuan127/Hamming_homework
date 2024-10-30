import json
import os
from helper_functions import call_hamming_and_transcribe
from llm_functions import determine_state, identify_speaker
from llm_parsers import parse_information, parse_question, parse_action
from llm_prompters import generate_next_prompt_question

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

# Print all responses from the business AI agent
print("\nBusiness AI Agent responses:")
with open("transcription_output.txt", "r") as f:
    lines = f.readlines()

information_database = []
questions_database = []
actions_database = []

for line in lines:
    if f"[Speaker {business_speaker}]" in line:
        # Extract just the text after the speaker tag
        response = line.split(f"[Speaker {business_speaker}]")[1].strip()
        print(f"- {response}")
        states = determine_state(GEMINI_API_KEY, GEMINI_MODEL, response)
        print(f"States: {states}\n")
        states = json.loads(states)
        for state in states:
            if state['state'] == 'information':
                parsed_info = parse_information(GEMINI_API_KEY, GEMINI_MODEL, state['text'], information_database)
                if parsed_info != "DUPLICATE":
                    information_database.append(parsed_info)    
            elif state['state'] == 'question' or state['state'] == 'action_request':
                parsed_question = parse_question(GEMINI_API_KEY, GEMINI_MODEL, state['text'], questions_database)
                if parsed_question != "DUPLICATE":
                    questions_database.append(parsed_question)
            elif state['state'] == 'action':
                parsed_action = parse_action(GEMINI_API_KEY, GEMINI_MODEL, state['text'], actions_database)
                if parsed_action != "DUPLICATE":
                    actions_database.append(parsed_action)

print(f"Information Database: {information_database}")
print(f"Questions Database: {questions_database}")
print(f"Actions Database: {actions_database}")