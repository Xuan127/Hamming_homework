# Hamming_homework

Justifications & Thought process
1. Saved API key and phone number in local environment for privacy
2. For transcription, it seems that the model just need to transcribe, so decided on Deepgram
3. Set Diarization to be true as there are 2 agents, saved conversation as .txt and .json for viewing, created another .json without "words" for easier viewing
4. Each agent is on a seperate channel, turn on multi-channel transciption, it will always work compared to diarization, since each agent has their own channels, diarization is not helpful, turning off diarization
