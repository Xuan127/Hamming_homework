import os, requests

# Define the API endpoint and authorization token
url = "https://app.hamming.ai/api/rest/exercise/start-call"
api_token = os.environ['HAMMING_API']
number_to_call = os.environ['NUMBER_TO_CALL']

# Set up the request headers with authorization
headers = {
    "Authorization": f"Bearer {api_token}",
    "Content-Type": "application/json"
}

# Define the request payload
data = {
    "phone_number": number_to_call,  # Replace with the actual phone number
    "prompt": "Say hi.",    # Replace with the desired system prompt
    "webhook_url": url   # Replace with your webhook URL
}

# Send the POST request
response = requests.post(url, headers=headers, json=data)

# Check the response status
if response.status_code == 200:
    print("Call started successfully:", response.json())
else:
    print("Failed to start call:", response.status_code, response.text)