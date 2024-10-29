import os, requests, json

# Define the API endpoint and authorization token
call_id = 'cm2ttm37h00gg95527ys7j2ig'
url = f"https://app.hamming.ai/api/media/exercise?id={call_id}"
api_token = os.environ['HAMMING_API']

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

# Define DeepGram API URL and authorization token
url = "https://api.deepgram.com/v1/listen?diarize=true&punctuate=true&utterances=true"
api_key = os.environ['DEEPGRAM_API']
audio_file_path = "call_recording.wav"

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
    with open("transcription_output.txt", "w") as txt_file:
        for utterance in utterances:
            speaker = utterance.get("speaker", "Unknown")
            transcript = utterance.get("transcript", "")
            txt_file.write(f"[Speaker: {speaker}] {transcript}\n")
    
    print("Transcription with speaker diarization saved to 'transcription_output.txt'")
else:
    print("Failed to transcribe audio:", response.status_code, response.text)