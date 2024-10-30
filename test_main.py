import json
import os
from helper_functions import call_per_node
from llm_prompters import generate_initial_prompt, generate_next_prompt
from conversation_graph import ConversationGraph

HAMMING_API_KEY = os.environ['HAMMING_API_KEY']
DEEPGRAM_API_KEY = os.environ['DEEPGRAM_API_KEY']
GEMINI_API_KEY = os.environ['GEMINI_API_KEY']
GEMINI_MODEL = "gemini-1.5-flash"
number_to_call = os.environ['NUMBER_TO_CALL']
business_description = "Air Conditioning and Plumbing"

initial_prompt = generate_initial_prompt(GEMINI_API_KEY, GEMINI_MODEL, business_description)

graph = ConversationGraph()
call_stacks = []

call_per_node(HAMMING_API_KEY, DEEPGRAM_API_KEY, GEMINI_API_KEY, GEMINI_MODEL, number_to_call, initial_prompt, graph, "start", '', call_stacks)
print(call_stacks)

while len(call_stacks) > 0:
    call_object = call_stacks.pop()
    print(call_object)
    print(call_stacks)
    next_prompt = generate_next_prompt(GEMINI_API_KEY, GEMINI_MODEL, business_description, call_object['question'], call_object['response'], graph.get_history(call_object['question']))
    print(next_prompt)
    call_per_node(HAMMING_API_KEY, DEEPGRAM_API_KEY, GEMINI_API_KEY, GEMINI_MODEL, number_to_call, next_prompt, graph, call_object['question'], call_object['response'], call_stacks)
    print(call_stacks)