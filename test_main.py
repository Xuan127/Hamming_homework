import os
from helpers import agent_call, retrieve_audio, transcribe_audio
import time

HAMMING_API_KEY = os.environ['HAMMING_API_KEY']
DEEPGRAM_API_KEY = os.environ['DEEPGRAM_API_KEY']
number_to_call = os.environ['NUMBER_TO_CALL']
prompt = "You are a customer calling an AI agent. Ask them what services they offer and how to sign up for a new account."

response = agent_call(HAMMING_API_KEY, number_to_call, prompt)
call_id = response.json()["id"]

audio_available = False
while not audio_available:
    time.sleep(5)
    print("Waiting for the audio to be available...")
    response = retrieve_audio(HAMMING_API_KEY, call_id)
    if response.status_code == 200:
        audio_available = True

transcribe_audio(DEEPGRAM_API_KEY, "call_recording.wav")
