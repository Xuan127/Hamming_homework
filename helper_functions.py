import requests, json, time, os
from conversation_graph import ConversationGraph
from helper_structs import ConversationState
from llm_functions import determine_state, identify_speaker, generate_question_response, check_in_history
from llm_parsers import parse_information, parse_question, parse_action

def agent_call(api_token, number_to_call, prompt):
    # Define the API endpoint and authorization token
    url = "https://app.hamming.ai/api/rest/exercise/start-call"

    # Set up the request headers with authorization
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }

    # Define the request payload
    data = {
        "phone_number": number_to_call,  # Replace with the actual phone number
        "prompt": prompt,
        "webhook_url": url   # Replace with your webhook URL
    }

    # Send the POST request
    response = requests.post(url, headers=headers, json=data)

    # Check the response status
    if response.status_code == 200:
        print("Call started successfully:", response.json())
    else:
        print("Failed to start call:", response.status_code, response.text)

    return response


def retrieve_audio(api_token, call_id):
    # Define the API endpoint and authorization token
    url = f"https://app.hamming.ai/api/media/exercise?id={call_id}"

    # Set up the request headers with authorization
    headers = {
        "Authorization": f"Bearer {api_token}"
    }

    # Send the GET request to retrieve the audio
    response = requests.get(url, headers=headers)

    # Check the response status
    if response.status_code == 200:
        # Save the audio file as a .wav file
        with open("call_recording.wav", "wb") as audio_file:
            audio_file.write(response.content)
        print("Audio file downloaded successfully as 'call_recording.wav'")
    else:
        print("Failed to retrieve audio file:", response.status_code, response.text)

    return response


def transcribe_audio(api_key, audio_file_path, save_as_txt=True, save_as_json=True, save_as_json_no_words=True):
    # Define DeepGram API URL and authorization token
    url = "https://api.deepgram.com/v1/listen?multichannel=true&punctuate=true&utterances=true&model=nova-2&smart_format=true"

    # Set up headers with the DeepGram API key and content type
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "audio/mp3"
    }

    # Open the audio file in binary mode and send the POST request
    with open(audio_file_path, "rb") as audio_file:
        response = requests.post(url, headers=headers, data=audio_file)

    # Check the response status
    if response.status_code == 200:
        # Parse the response JSON to extract speaker and transcript data
        utterances = response.json().get("results", {}).get("utterances", [])
        
        # Write the formatted output to a text file
        if save_as_txt:
            with open("transcription_output.txt", "w") as txt_file:
                for utterance in utterances:
                    channel = utterance.get("channel", "Unknown")
                    transcript = utterance.get("transcript", "")
                    txt_file.write(f"[Speaker {channel}] {transcript}\n")

            print("Transcription with speaker diarization saved to 'transcription_output.txt'")

        # Save the utterances data to a JSON file
        if save_as_json:
            with open("transcription_output.json", "w") as json_file:
                json.dump(utterances, json_file, indent=4)

            print("Transcription data saved to 'transcription_output.json'")

        # Save the utterances data to a JSON file without the "words" key
        if save_as_json_no_words:
            # Create a new list excluding the "words" key from each utterance
            utterances_without_words = [
                {key: value for key, value in utterance.items() if key != "words"}
                for utterance in utterances
            ]
        
            # Save the filtered utterances to a new JSON file
            with open("transcription_output_no_words.json", "w") as json_no_words_file:
                json.dump(utterances_without_words, json_no_words_file, indent=4)
            
            print("Transcription data without 'words' saved to 'transcription_output_no_words.json'")
    else:
        print("Failed to transcribe audio:", response.status_code, response.text)

    return response


def call_gemini(api_key, prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={api_key}"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes
    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")
    
    return response

def call_hamming_and_transcribe(hamming_api_key, deepgram_api_key, number_to_call, initial_prompt):
    response = agent_call(hamming_api_key, number_to_call, initial_prompt)
    call_id = response.json()["id"]

    time.sleep(30)
    audio_available = False
    while not audio_available:
        time.sleep(10)
        print("Waiting for the audio to be available...")
        response = retrieve_audio(hamming_api_key, call_id)
        if response.status_code == 200:
            audio_available = True

    transcribe_audio(deepgram_api_key, "call_recording.wav")

def call_per_node(hamming_api_key, deepgram_api_key, gemini_api_key, gemini_model, number_to_call, prompt, graph, node, condition, call_stacks):
    call_hamming_and_transcribe(hamming_api_key, deepgram_api_key, number_to_call, prompt)

    # Read the transcription output
    with open("transcription_output.txt", "r") as f:
        transcript = f.read()

    business_speaker = identify_speaker(gemini_api_key, gemini_model, transcript)

    print(f"\nIdentified business AI agent as Speaker {business_speaker}")

    # Print all responses from the business AI agent
    print("\nBusiness AI Agent responses:")
    with open("transcription_output.txt", "r") as f:
        lines = f.readlines()

    questions_database = []
    actions_database = []

    for line in lines:
        if f"[Speaker {business_speaker}]" in line:
            # Extract just the text after the speaker tag
            response = line.split(f"[Speaker {business_speaker}]")[1].strip()
            print(f"- {response}")
            states = determine_state(gemini_api_key, gemini_model, response)
            print(f"States: {states}\n")
            states = json.loads(states)
            for state in states:
                if state['state'] == 'information':
                    parsed_info = parse_information(gemini_api_key, gemini_model, state['text'], graph.information_database)
                    if parsed_info != "DUPLICATE":
                        graph.information_database.append(parsed_info)    
                elif state['state'] == 'question' or state['state'] == 'action_request':
                    parsed_question = parse_question(gemini_api_key, gemini_model, state['text'], questions_database)
                    if parsed_question != "DUPLICATE":
                        questions_database.append(parsed_question)
                elif state['state'] == 'action' or state['state'] == 'transfer':
                    parsed_action = parse_action(gemini_api_key, gemini_model, state['text'], actions_database)
                    if parsed_action != "DUPLICATE":
                        actions_database.append(parsed_action)

    print(f"Information Database: {graph.information_database}")
    print(f"Questions Database: {questions_database}")
    print(f"Actions Database: {actions_database}")

    history = list(graph.get_history(node))
    history.append({'question': node, 'response': condition})

    list_of_responses = []
    for question in questions_database:
        if not check_in_history(gemini_api_key, gemini_model, history, question):
            list_of_responses = generate_question_response(gemini_api_key, gemini_model, question, graph.information_database)
            list_of_responses = json.loads(list_of_responses)
            break

    if len(list_of_responses) > 0:
        graph.add_node_with_edge(node, list_of_responses[0]['question'], ConversationState.QUESTION, condition, history)
        for response in list_of_responses:
            new_response = {'question': response['question'], 'response': response['response']}
            call_stacks.append(new_response)
    elif len(actions_database) > 0:
        for action in actions_database:
            graph.add_node_with_edge(node, action, ConversationState.ACTION, condition, history)

    # graph.visualize_graph()

if __name__ == "__main__":
    hamming_api_key = os.getenv("HAMMING_API_KEY")
    deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
    number_to_call = os.getenv("NUMBER_TO_CALL")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    gemini_model = "gemini-1.5-flash"
    graph = ConversationGraph()
    node = "start"
    call_stacks = []
    prompt = """You are a caller testing the business's AI system. 
        Say that you are an existing customer and ask for information about the business. 
        Say that you name is John Doe and your address is 123 Main Street.
        Say that your aircon broke down and you need help."""
    call_per_node(hamming_api_key, deepgram_api_key, gemini_api_key, gemini_model, number_to_call, prompt, graph, node, call_stacks)
