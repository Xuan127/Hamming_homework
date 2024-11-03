import logging, requests, json, time, os, datetime
from typing import Optional
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(f"logs/helper_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        # logging.StreamHandler()
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

    logger.debug(f"Initiating call to {number_to_call}")
    print(f"Initiating call to {number_to_call}")
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        logger.info(f"Call started successfully: {response.json()}")
        print(f"Call started successfully: {response.json()}")
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
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            with open(f"transcription_output_{timestamp}.txt", "w") as txt_file:
                for utterance in utterances:
                    channel = utterance.get("channel", "Unknown")
                    transcript = utterance.get("transcript", "")
                    txt_file.write(f"[Speaker {channel}] {transcript}\n")
            with open(f"transcription_output.txt", "w") as txt_file:
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

        print("transcription successful")
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
    logger.debug(f"call_hamming_and_transcribe - Parameters: hamming_api_key=<hidden>, "
                 f"deepgram_api_key=<hidden>, number_to_call={number_to_call}, initial_prompt=<hidden>")
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
    max_retries = 600  # Total wait time: 30 + (10 * 600) = 6300 seconds
    retries = 0

    while not audio_available and retries < max_retries:
        time.sleep(10)
        retries += 1
        logger.info("Checking if audio is available...")
        print("Checking if audio is available...")
        response = retrieve_audio(hamming_api_key, call_id)
        if response and response.status_code == 200:
            audio_available = True
            logger.info("Audio is now available for transcription.")
            print("Audio is now available for transcription.")
        else:
            logger.info("Audio not yet available. Continuing to wait...")
            print("Audio not yet available. Continuing to wait...")

    if not audio_available:
        logger.error("Audio not available after multiple attempts. Aborting transcription process.")
        return

    transcription = transcribe_audio(deepgram_api_key, f"call_recording.wav", save_as_txt=True, save_as_json=False, save_as_json_no_words=False)
    if transcription:
        logger.info("Transcription completed successfully.")
        print("Transcription completed successfully.")
    else:
        logger.error("Transcription failed.")

def prompt_creator(api_key: str, model_name: str, business_description: str, nodes: list[dict], edges: list[dict]) -> str:
    """
    Creates a system prompt for an AI Voice Agent to test business conversations.

    Args:
        api_key (str): OpenAI API key
        model_name (str): Name of the OpenAI model to use
        business_description (str): Description of the business being tested
        nodes (list[dict]): List of existing conversation nodes
        edges (list[dict]): List of existing conversation edges/paths

    Returns:
        str: Generated system prompt for the AI Voice Agent

    Raises:
        ValueError: If required parameters are missing or invalid
        Exception: For OpenAI API errors or other unexpected issues
    """
    logger.info("Generating system prompt for AI Voice Agent")
    print("Generating system prompt for AI Voice Agent")
    logger.debug(f"Parameters: model_name={model_name}, business_description={business_description}, "
                f"nodes_count={len(nodes)}, edges_count={len(edges)}")

    if not api_key:
        logger.error("Missing OpenAI API key")
        raise ValueError("OpenAI API key is required")

    if not business_description:
        logger.error("Missing business description")
        raise ValueError("Business description is required")

    try:
        client = OpenAI(api_key=api_key)
        system_instruction = f"""
            <instructions>
            You are a prompt engineer specializing in creating system prompts for AI Voice Agents that will call and test businesses.
            The agent you are prompting for will be the caller, initiating conversations with the business to test their AI system.
            The business description is: {business_description}
            Create a comprehensive system prompt that will help test all possible conversation paths and scenarios.
            You are given the nodes and edges of the current conversation tree, use them to assign tasks to the caller agent.
            </instructions>

            <prompt requirements>
            The prompt should:
            1. Define the agent's role as a caller testing the business's AI system
            2. Specify various test scenarios to try (based on the nodes and edges)
            3. You do not need to repeat the existing scenarios from the nodes and edges, just add more.
            4. For each decision node, explore different responses that is not explored according to the edges
            5. The caller agent should not disclose that it is a tester agent, it should not say that it is testing the business's AI system.
            6. The prompt should be in markdown format.

            Format the response as a clear, structured system prompt that can be used directly with an AI model.
            Do not talk to me at all.
            </prompt requirements>

            <current decision tree>
            the nodes are: {nodes}
            the edges are: {edges}
            </current decision tree>
        """

        logger.debug("Sending request to OpenAI API")
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": system_instruction}]
        )
        
        logger.info("Successfully generated system prompt")
        # Save the generated prompt to a file
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(f"system_prompt_{timestamp}.txt", "w") as f:
            f.write(response.choices[0].message.content)
    
        print("prompt created")
        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"Error generating system prompt: {str(e)}")
        raise Exception(f"Failed to generate system prompt: {str(e)}")

if __name__ == "__main__":
    openai_api_key = os.getenv("OPENAI_API_KEY")
    business_description = "aircon servicing"

    print(prompt_creator(openai_api_key, "o1-preview", business_description, [], []))
