import os, requests, json

# Define DeepGram API URL and authorization token
url = "https://api.deepgram.com/v1/listen?multichannel=true&punctuate=true&utterances=true&model=nova-2&smart_format=true"
api_key = os.environ['DEEPGRAM_API']
audio_file_path = "call_recording.wav"
save_as_txt = True
save_as_json = True
save_as_json_no_words = True

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