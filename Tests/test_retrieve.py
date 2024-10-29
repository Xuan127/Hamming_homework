import os, requests

# Define the API endpoint and authorization token
call_id = 'cm2tvwpec00kaq68au8liuba4'
url = f"https://app.hamming.ai/api/media/exercise?id={call_id}"
api_token = os.environ['HAMMING_API_KEY']

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