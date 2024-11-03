from DecisionTree import DecisionTree
from helpers import call_hamming_and_transcribe, prompt_creator
from tree_helpers import parse_nodes_and_edges, get_nodes, get_edges, parse_tree
import os, datetime
import streamlit as st

hamming_api_key = os.environ.get("HAMMING_API_KEY")
deepgram_api_key = os.environ.get("DEEPGRAM_API_KEY")
openai_api_key = os.environ.get("OPENAI_API_KEY")
number_to_call = os.environ.get("NUMBER_TO_CALL")
business_description = "Air Conditioning and Plumbing Company"
st.set_page_config(layout="wide")
tree = DecisionTree()
nodes = []
edges = []

while True:
    print("looping")
    prompt = prompt_creator(openai_api_key, "o1-preview", business_description, nodes, edges)
    call_hamming_and_transcribe(hamming_api_key, deepgram_api_key, number_to_call, prompt)
    conversation = open("transcription_output.txt", "r").read()
    text = parse_nodes_and_edges(openai_api_key, "o1-preview", conversation, nodes, edges)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f"parsed_text_output_{timestamp}.txt", "w") as f:
        f.write(str(text))
    new_nodes = get_nodes(openai_api_key, "gpt-4o", text)
    new_edges = get_edges(openai_api_key, "gpt-4o", text)
    print('new_nodes', new_nodes)
    print('new_edges', new_edges)
    if new_nodes == None and new_edges == None:
        break
    if new_nodes == None: new_nodes = []
    if new_edges == None: new_edges = []
    nodes = nodes + new_nodes
    edges = edges + new_edges
    tree = parse_tree(tree, new_nodes, new_edges)
    tree.display()
print('DONE')

