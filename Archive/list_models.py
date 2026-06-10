import requests
import os
import json

with open(r"C:\Users\Irak\Desktop\AI_Agent\DigitalHistory\groqapi.txt", "r") as f:
    api_key = f.read().strip().replace('"', '').replace(',', '')

headers = {
    'Authorization': 'Bearer ' + api_key
}

try:
    response = requests.get('https://api.groq.com/openai/v1/models', headers=headers)
    if response.status_code == 200:
        models = response.json().get('data', [])
        print("Available models:")
        for m in models:
            print("-", m.get('id'))
    else:
        print("Error fetching models:", response.status_code, response.text)
except Exception as e:
    print("Exception:", e)
