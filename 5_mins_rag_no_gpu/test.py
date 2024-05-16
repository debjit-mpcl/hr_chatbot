import requests

url = 'http://mixtral-8x7b-instruct-v0-1:9099/v1/chat/completions'
headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json'
}

data = {
    "model": "mixtral-8x7b-instruct-v0-1",
    "messages": [
        {"role": "user", "content": "Hello there how are you?"},
        {"role": "assistant", "content": "Good and you?"},
        {"role": "user", "content": "Write a short note on naxalism ?"}
    ],
    "max_tokens": 1024,
    "top_p": 1,
    "n": 1,
    "stream": False,
    "stop": "string",
    "frequency_penalty": 0.0
}

response = requests.post(url, json=data, headers=headers)

# Process the response as needed
for c in response.json()['choices']:
  print(c['message']['content'])
