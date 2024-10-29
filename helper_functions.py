import requests, json

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