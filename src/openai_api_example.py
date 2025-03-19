import requests

API_KEY = "sk-proj-gCQDRtwQ7chS_4w6Su-m6hPl2eanCSOn|3Lpsz gKdBh4_Sk7jr9qE__tfqSjeYVU7k32SDidkfT3Blb kFJKO1MtLeeWKJtE9_eUlJMCvSpCODy57QJ84 y38zalXZ5shHxe7skG1DezOx3xLCY29nyqeFf8w"
URL = "https://api.openai.com/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

data = {
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello, how are you?"}]
}

response = requests.post(URL, headers=headers, json=data)

print(response.json())
