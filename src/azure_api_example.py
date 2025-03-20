import os
from dotenv import load_dotenv
import requests
import itertools

def test_api_key(key, region="switzerlandnorth"):
    """Test if an API key is valid by making a request to Azure Speech Service."""
    endpoint = "https://switzerlandnorth.api.cognitive.microsoft.com/"
    headers = {"Ocp-Apim-Subscription-Key": key.strip()}
    try:
        response = requests.get(endpoint, headers=headers)
        return response.status_code == 200
    except:
        return False

# Load environment variables
load_dotenv()

# Get API key from environment
api_key = os.getenv("AZURE_SPEECH_KEY")

if not api_key:
    print("❌ AZURE_SPEECH_KEY not found in .env file!")
    exit(1)

print(f"Original API key: {api_key}")

# Find positions of 'l' and 'I' in the API key
ambiguous_positions = [i for i, char in enumerate(api_key) if char in 'lI']
print(f"\nFound {len(ambiguous_positions)} ambiguous characters at positions: {ambiguous_positions}")

if not ambiguous_positions:
    print("No ambiguous 'l' or 'I' characters found in the API key.")
    exit(0)

# Generate all possible combinations
combinations = list(itertools.product(['l', 'I'], repeat=len(ambiguous_positions)))
print(f"Testing {len(combinations)} possible combinations...")

# Try each combination
for combo in combinations:
    test_key = list(api_key)
    for pos, char in zip(ambiguous_positions, combo):
        test_key[pos] = char
    test_key = ''.join(test_key)
    
    print(f"\nTrying key: {test_key}")
    if test_api_key(test_key):
        print("✅ Found working API key!")
        print(f"Working key: {test_key}")
        print("\nPositions that were changed:")
        for pos, char in zip(ambiguous_positions, combo):
            print(f"Position {pos}: Changed to '{char}'")
        exit(0)

# No working combination found
print("\n❌ No working combination found.")
