# Hamming_homework

Justifications & Thought process
1. Saved API key and phone number in local environment for privacy
2. For transcription, it seems that the model just need to transcribe, so decided on Deepgram
3. Set Diarization to be true as there are 2 agents, saved conversation as .txt and .json for viewing, created another .json without "words" for easier viewing
4. Each agent is on a seperate channel, turn on multi-channel transciption, it will always work compared to diarization, since each agent has their own channels, diarization is not helpful, turning off diarization
5. Created helper functions, to make code neater and abstracts functions
6. Tried to catch webhook payload with flask, but couldn't get it to work, will just repeatedly call the get request until response code 200
7. Seems like the given Hamming API does not allow streaming, so there is only one chance to prompt the tester AI before the call. We will need to design a prompt engineering AI to create prompt and interatively explore the othe agent
8. Using Gemini API for the prompt engineering AI, since its API is free. Used the gemini-1.5-pro model, which is more powerful
9. Created a conversation graph to visualize the conversation flow

Extra:
- Looked at Hamming APIs but realized that I have no access, keep to the 2 APIs given
- Tried to explore network packets of the API calls to discover more query parameters, did not discover anything
- Looked up tools such as Postman and Fiddler for API exploration but decided that it was not neccesary
- Looked at Hamming github, there are no examples of these 2 APIs, so not very helpful

Todo:
- Finetune Gemini API
- Fine-tune Deepgram API
- different language
- reaction to profanities
- logging
- error handling

