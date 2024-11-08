# Hamming_homework

To run,
1. Install dependencies: pip install -r requirements.txt
2. Run the script: streamlit run main.py

Justifications & Thought process
1. Saved API key and phone number in local environment for privacy
2. For transcription, it seems that the model just need to transcribe, so decided on Deepgram
3. Set Diarization to be true as there are 2 agents, saved conversation as .txt and .json for viewing, created another .json without "words" for easier viewing
4. Each agent is on a seperate channel, turn on multi-channel transciption, it will always work compared to diarization, since each agent has their own channels, diarization is not helpful, turning off diarization
5. Created helper functions, to make code neater and abstracts functions
6. Tried to catch webhook payload with flask, but couldn't get it to work, will just repeatedly call the get request until response code 200
7. Created a conversation graph to visualize the conversation flow
8. Seems like the given Hamming API does not allow streaming, so there is only one chance to prompt the tester AI before the call. We will need to design a prompt engineering AI to create prompt and interatively explore the othe agent
9. Using Gemini API for the prompt engineering AI, since its API is free. Used the gemini-1.5-pro model, which is more powerful, using the SDK as it is easier to use
10. Created a function to generate the initial prompt for the tester AI
11. Created a function to identify the state of the conversation, to be used for crafting the next prompt
12. Added safety settings to the Gemini API, so that it will not flag as harmful
13. Created a function to identify the business AI agent
14. Added sleep to the LLM functions, so that it will not be too quick to respond
15. Used different functions for each LLM call, so that it is easier to debug and specialize
16. Created a function to generate the next prompt for the question
17. Redo the logic, will parse the entire conversation at once now
18. Go back to previous logic. Added a function to check if a question has been asked before, to avoid repeating the same question
19. Created working flow that can draw a graph.
20. Added a function to generate the next prompts
21. Tried multiprocessing, doesnt seem to work
22. Added logging and error handling
23. Added sample prompts and transcript for testing
24. Tried groq, doesn't work well either
25. Tried OpenAI API, works much better, parsing the entire conversation at once now

Extra:
- Could just use gemini to generate the entire graph, but decided against it as it is not very robust
- Looked at Hamming APIs but realized that I have no access, keep to the 2 APIs given
- Tried to explore network packets of the API calls to discover more query parameters for the APIs, did not discover anything
- Looked up tools such as Postman and Fiddler for API exploration but decided that it was not neccesary
- Looked at Hamming github, there are no examples of these 2 APIs, so not very helpful

Future improvements:
- Finetune Gemini prompts
- Finetune Deepgram API
- different language
- reaction to profanities