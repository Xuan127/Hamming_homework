import logging, requests, json, time, os, multiprocessing
from typing import Optional

from deprecated.conversation_graph import ConversationGraph
from deprecated.helper_structs import ConversationState
from deprecated.llm_functions import (
    determine_state,
    identify_speaker,
    generate_question_response,
    check_in_history,
)
from deprecated.llm_parsers import parse_information, parse_question, parse_action

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/helper_functions.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def agent_call(api_token: str, number_to_call: str, prompt: str) -> Optional[requests.Response]:
    """
    Initiates a call using the Hamming API.

    Parameters:
        api_token (str): Bearer token for authorization.
        number_to_call (str): The phone number to call.
        prompt (str): The prompt to be used in the call.

    Returns:
        Optional[requests.Response]: The response from the API call if successful, else None.
    """
    url = "https://app.hamming.ai/api/rest/exercise/start-call"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    data = {
        "phone_number": number_to_call,
        "prompt": prompt,
        "webhook_url": url  # This might need to be another endpoint
    }

    logger.info(f"Initiating call to {number_to_call}")
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        logger.info(f"Call started successfully: {response.json()}")
        return response
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred while starting call: {http_err} - Response: {response.text}")
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Request exception occurred while starting call: {req_err}")
    except Exception as err:
        logger.error(f"An unexpected error occurred while starting call: {err}")
    return None

def retrieve_audio(api_token: str, call_id: str) -> Optional[requests.Response]:
    """
    Retrieves the audio recording of a call using the Hamming API.

    Parameters:
        api_token (str): Bearer token for authorization.
        call_id (str): The unique identifier of the call.

    Returns:
        Optional[requests.Response]: The response containing audio content if successful, else None.
    """
    url = f"https://app.hamming.ai/api/media/exercise?id={call_id}"
    headers = {
        "Authorization": f"Bearer {api_token}"
    }

    logger.info(f"Retrieving audio for call ID: {call_id}")
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        with open("call_recording.wav", "wb") as audio_file:
            audio_file.write(response.content)
        logger.info("Audio file downloaded successfully as 'call_recording.wav'")
        return response
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred while retrieving audio: {http_err} - Response: {response.text}")
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Request exception occurred while retrieving audio: {req_err}")
    except Exception as err:
        logger.error(f"An unexpected error occurred while retrieving audio: {err}")
    return None

def transcribe_audio(
    api_key: str,
    audio_file_path: str,
    save_as_txt: bool = True,
    save_as_json: bool = True,
    save_as_json_no_words: bool = True
) -> Optional[dict]:
    """
    Transcribes audio using the DeepGram API and saves the results in various formats.

    Parameters:
        api_key (str): DeepGram API key.
        audio_file_path (str): Path to the audio file to transcribe.
        save_as_txt (bool): Whether to save the transcription as a text file.
        save_as_json (bool): Whether to save the entire transcription as a JSON file.
        save_as_json_no_words (bool): Whether to save the transcription sans 'words' key as a JSON file.

    Returns:
        Optional[dict]: JSON response from DeepGram if successful, else None.
    """
    url = (
        "https://api.deepgram.com/v1/listen?"
        "multichannel=true&punctuate=true&utterances=true&"
        "model=nova-2&smart_format=true"
    )
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "audio/mp3"
    }

    logger.info(f"Starting transcription for file: {audio_file_path}")
    try:
        with open(audio_file_path, "rb") as audio_file:
            response = requests.post(url, headers=headers, data=audio_file)
        response.raise_for_status()
        data = response.json()
        utterances = data.get("results", {}).get("utterances", [])

        if save_as_txt:
            with open("transcription_output.txt", "w") as txt_file:
                for utterance in utterances:
                    channel = utterance.get("channel", "Unknown")
                    transcript = utterance.get("transcript", "")
                    txt_file.write(f"[Speaker {channel}] {transcript}\n")
            logger.info("Transcription with speaker diarization saved to 'transcription_output.txt'")

        if save_as_json:
            with open("transcription_output.json", "w") as json_file:
                json.dump(utterances, json_file, indent=4)
            logger.info("Transcription data saved to 'transcription_output.json'")

        if save_as_json_no_words:
            utterances_without_words = [
                {key: value for key, value in utterance.items() if key != "words"}
                for utterance in utterances
            ]
            with open("transcription_output_no_words.json", "w") as json_no_words_file:
                json.dump(utterances_without_words, json_no_words_file, indent=4)
            logger.info("Transcription data without 'words' saved to 'transcription_output_no_words.json'")

        return data
    except FileNotFoundError:
        logger.error(f"Audio file not found: {audio_file_path}")
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred during transcription: {http_err} - Response: {response.text}")
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Request exception occurred during transcription: {req_err}")
    except json.JSONDecodeError as json_err:
        logger.error(f"JSON decode error: {json_err}")
    except Exception as err:
        logger.error(f"An unexpected error occurred during transcription: {err}")
    return None

def call_gemini(api_key: str, prompt: str) -> Optional[requests.Response]:
    """
    Calls the Gemini API to generate content based on the provided prompt.

    Parameters:
        api_key (str): Gemini API key.
        prompt (str): The prompt text to generate content from.

    Returns:
        Optional[requests.Response]: The response from the Gemini API if successful, else None.
    """
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

    logger.info("Making request to Gemini API")
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        logger.info("Gemini API call successful")
        return response
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred during Gemini API call: {http_err} - Response: {response.text}")
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Request exception occurred during Gemini API call: {req_err}")
    except Exception as err:
        logger.error(f"An unexpected error occurred during Gemini API call: {err}")
    return None

def call_hamming_and_transcribe(
    hamming_api_key: str,
    deepgram_api_key: str,
    number_to_call: str,
    initial_prompt: str
) -> None:
    """
    Orchestrates the process of making a call via Hamming, retrieving the audio, and transcribing it.

    Parameters:
        hamming_api_key (str): Hamming API key.
        deepgram_api_key (str): DeepGram API key.
        number_to_call (str): The phone number to call.
        initial_prompt (str): The initial prompt for the call.

    Returns:
        None
    """
    logger.info("Starting Hamming call and transcription process")
    response = agent_call(hamming_api_key, number_to_call, initial_prompt)
    if not response:
        logger.error("Failed to initiate call. Aborting transcription process.")
        return

    call_id = response.json().get("id")
    if not call_id:
        logger.error("Call ID not found in response. Aborting transcription process.")
        return

    logger.info(f"Call initiated with ID: {call_id}. Waiting for audio to become available...")
    time.sleep(30)  # Initial wait before checking for audio
    audio_available = False

    while not audio_available:
        time.sleep(10)
        logger.info("Checking if audio is available...")
        response = retrieve_audio(hamming_api_key, call_id)
        if response and response.status_code == 200:
            audio_available = True
            logger.info("Audio is now available for transcription.")
        else:
            logger.info("Audio not yet available. Continuing to wait...")

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

HAMMING_API_KEY = os.getenv("HAMMING_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-1.5-pro"
NUMBER_TO_CALL = os.getenv("NUMBER_TO_CALL")

def test_call_hamming_and_transcribe():
    prompt = open("examples/sample_next_prompt_3.txt", "r").read()
    call_hamming_and_transcribe(HAMMING_API_KEY, DEEPGRAM_API_KEY, NUMBER_TO_CALL, prompt)

def test_call_hamming_and_transcribe_multiprocessing():
    prompt_1 = open("examples/sample_next_prompt_1.txt", "r").read()
    prompt_2 = open("examples/sample_next_prompt_2.txt", "r").read()
    processes = []
    processes.append(multiprocessing.Process(target=call_hamming_and_transcribe, args=(HAMMING_API_KEY, DEEPGRAM_API_KEY, NUMBER_TO_CALL, prompt_1)))
    processes.append(multiprocessing.Process(target=call_hamming_and_transcribe, args=(HAMMING_API_KEY, DEEPGRAM_API_KEY, NUMBER_TO_CALL, prompt_2)))

    for process in processes:
        process.start()
    for process in processes:
        process.join()

if __name__ == "__main__":
    test_call_hamming_and_transcribe()
