import requests

url = 'http://rag_embedding_model:8006/v1/embeddings'
headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json'
}

data = {
    "model": "NV-Embed-QA",
    "input":["Hello world"],
    "input_type":"query"
}

response = requests.post(url, json=data, headers=headers)

print(response.json()['data'][0]['embedding'])
