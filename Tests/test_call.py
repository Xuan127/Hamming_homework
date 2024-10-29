import os, requests

# Define the API endpoint and authorization token
url = "https://app.hamming.ai/api/rest/exercise/start-call"
api_token = os.environ['HAMMING_API_KEY']
number_to_call = os.environ['NUMBER_TO_CALL']
prompt = "You are a customer calling an AI agent. Ask them what services they offer and how to sign up for a new account."

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