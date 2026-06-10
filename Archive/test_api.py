import requests
import os

with open(r"C:\Users\Irak\Desktop\AI_Agent\DigitalHistory\groqapi.txt", "r") as f:
    api_key = f.read().strip().replace('"', '').replace(',', '')

data = {
    "model": "openai/gpt-oss-120b",
    "messages": [{"role": "user", "content": "hello"}]
}

headers = {
    'Authorization': 'Bearer ' + api_key,
    'Content-Type': 'application/json'
}

try:
    response = requests.post('https://api.groq.com/openai/v1/chat/completions', json=data, headers=headers)
    print("Status Code:", response.status_code)
    print("Response:", response.text)
except Exception as e:
    print("Error:", e)
