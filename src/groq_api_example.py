
import os
from groq import Groq

client = Groq(api_key="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")
filename = os.path.dirname(__file__) + "/audio.m4a"

with open(filename, "rb") as file:
    transcription = client.audio.transcriptions.create(
      file=(filename, file.read()),
      model="whisper-large-v3",
      response_format="verbose_json",
    )
    print(transcription.text)
      