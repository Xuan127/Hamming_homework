import json
import os
from helper_functions import call_hamming_and_transcribe, call_per_node
from llm_functions import determine_state, identify_speaker, generate_question_response, check_in_history
from llm_parsers import parse_information, parse_question, parse_action
from conversation_graph import ConversationGraph
from helper_structs import ConversationState

HAMMING_API_KEY = os.environ['HAMMING_API_KEY']
DEEPGRAM_API_KEY = os.environ['DEEPGRAM_API_KEY']
GEMINI_API_KEY = os.environ['GEMINI_API_KEY']
GEMINI_MODEL = "gemini-1.5-flash"
number_to_call = os.environ['NUMBER_TO_CALL']
business_description = "Air Conditioning and Plumbing"

# initial_prompt = generate_initial_prompt(GEMINI_API_KEY, GEMINI_MODEL, business_description)

graph = ConversationGraph()
call_stacks = []

prompt = """You are a caller testing the business's AI system. 
        Say that you are an existing customer and ask for information about the business. 
        Say that you name is John Doe and your address is 123 Main Street.
        Say that your aircon broke down and you need help."""

call_per_node(HAMMING_API_KEY, DEEPGRAM_API_KEY, GEMINI_API_KEY, GEMINI_MODEL, number_to_call, prompt, graph, "start", '', call_stacks)
print(call_stacks)

while len(call_stacks) > 0:
    call_object = call_stacks.pop()
    print(call_object)
    print(call_stacks)

    call_per_node(HAMMING_API_KEY, DEEPGRAM_API_KEY, GEMINI_API_KEY, GEMINI_MODEL, number_to_call, prompt, graph, call_object['question'], call_object['response'], call_stacks)
    print(call_stacks)