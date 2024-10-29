import requests
import json
import os

def call_gemini_api(api_key, prompt):
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

if __name__ == "__main__":
    API_KEY = os.environ['GEMINI_API_KEY']
    prompt = "Explain how AI works"
    
    result = call_gemini_api(API_KEY, prompt)
    if result:
        print(json.dumps(result, indent=2))