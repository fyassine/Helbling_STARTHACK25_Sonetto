import uuid
import json
import os
import azure.cognitiveservices.speech as speechsdk
import numpy as np
import soundfile as sf
import io
import tempfile
import time

from flask import Flask, request, jsonify
from flask_sock import Sock
from flask_cors import CORS
from flasgger import Swagger

from groq import Groq
from dotenv import load_dotenv

load_dotenv()
AZURE_SPEECH_KEY = os.environ.get("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = "switzerlandnorth"
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Handle HTTP requests & responses
app = Flask(__name__) 

# Handle WebSocket connections for real-time bidirectional communication between client & server
# This is used for sending speech-to-text results back to clients in real-time
sock = Sock(app) 

# Enable Cross-Origin Resource Sharing (CORS) for the app 
# This allows our API to be accessed from different domains/origins
# Essential if the frontend is hosted on a different domain than the backend
cors = CORS(app)

# Initialize Swagger for API documentation
# This generates API documentation based on the docstrings in the code
swagger = Swagger(app)

sessions = {}

def transcribe_whisper(audio_recording):
    audio_file = io.BytesIO(audio_recording)
    audio_file.name = 'audio.wav'  # Whisper requires a filename with a valid extension
    transcription = client.audio.transcriptions.create(
        model="whisper-large-v3",
        file=audio_file,
        #language = ""  # specify Language explicitly
    )
    print(f"openai transcription: {transcription.text}")
    return transcription.text
    
# def transcribe_preview(session):
#     if session["audio_buffer"] is not None:
#         text = transcribe_whisper(session["audio_buffer"])
#         # send transcription
#         ws = session.get("websocket")
#         if ws:
#             message = {
#                 "event": "recognizing",
#                 "text": text,
#                 "language": session["language"]
#             }
#             ws.send(json.dumps(message))

@app.route("/chats/<chat_session_id>/sessions", methods=["POST"])
def open_session(chat_session_id):
    """
    Open a new voice input session and start continuous recognition.
    ---
    tags:
      - Sessions
    parameters:
      - name: chat_session_id
        in: path
        type: string
        required: true
        description: ID of the chat session
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - language
          properties:
            language:
              type: string
              description: Language code for speech recognition (e.g., en-US)
    responses:
      200:
        description: Session created successfully
        schema:
          type: object
          properties:
            session_id:
              type: string
              description: Unique identifier for the voice recognition session
      400:
        description: Language parameter missing
        schema:
          type: object
          properties:
            error:
              type: string
              description: Description of the error
    """
    session_id = str(uuid.uuid4())

    body = request.get_json()
    if "language" not in body:
        return jsonify({"error": "Language not specified"}), 400
    language = body["language"]

    sessions[session_id] = {
        "audio_buffer": None,
        "chatSessionId": chat_session_id,
        "language": language,
        "websocket": None  # will be set when the client connects via WS (WebSocket)
    }

    return jsonify({"session_id": session_id})


@app.route("/chats/<chat_session_id>/sessions/<session_id>/wav", methods=["POST"])
def upload_audio_chunk(chat_session_id, session_id):
    """
    Upload an audio chunk (expected 16kb, ~0.5s of WAV data).
    The chunk is appended to the push stream for the session.
    ---
    tags:
      - Sessions
    parameters:
      - name: chat_session_id
        in: path
        type: string
        required: true
        description: ID of the chat session
      - name: session_id
        in: path
        type: string
        required: true
        description: ID of the voice input session
      - name: audio_chunk
        in: body
        required: true
        schema:
          type: string
          format: binary
          description: Raw WAV audio data
    responses:
      200:
        description: Audio chunk received successfully
        schema:
          type: object
          properties:
            status:
              type: string
              description: Status message
      404:
        description: Session not found
        schema:
          type: object
          properties:
            error:
              type: string
              description: Description of the error
    """
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404

    audio_data = request.get_data()  # raw binary data from the POST body

    if sessions[session_id]["audio_buffer"] is not None:
        sessions[session_id]["audio_buffer"] = sessions[session_id]["audio_buffer"] + audio_data
    else:
        sessions[session_id]["audio_buffer"] = audio_data

    # TODO optionally transcribe real time audio chunks, see transcribe_preview()

    return jsonify({"status": "audio_chunk_received"})


@app.route("/chats/<chat_session_id>/sessions/<session_id>", methods=["DELETE"])
def close_session(chat_session_id, session_id):
    """
    Close the session (stop recognition, close push stream, cleanup).
    
    ---
    tags:
      - Sessions
    parameters:
      - name: chat_session_id
        in: path
        type: string
        required: true
        description: The ID of the chat session
      - name: session_id
        in: path
        type: string
        required: true
        description: The ID of the session to close
    responses:
      200:
        description: Session successfully closed
        schema:
          type: object
          properties:
            status:
              type: string
              example: session_closed
      404:
        description: Session not found
        schema:
          type: object
          properties:
            error:
              type: string
              example: Session not found
    """
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404

    if sessions[session_id]["audio_buffer"] is not None:
        # Process audio for speaker diarization
        audio = sessions[session_id]["audio_buffer"]
        
        # Convert audio buffer to WAV format for Azure Speech SDK
        audio_data = np.frombuffer(audio, dtype=np.int16)
        
        temp_dir = tempfile.gettempdir()
        temp_file = os.path.join(temp_dir, f"audio_{session_id}.wav")
        
        sf.write(temp_file, audio_data, 16000, format='WAV')
        
        try:
            # Initialize Azure Speech config
            speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
            print(f"Using Azure Speech Service with region: {AZURE_SPEECH_REGION}")
            
            # Create audio config from the temporary file
            audio_config = speechsdk.audio.AudioConfig(filename=temp_file)
            
            # Create speech recognizer
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config,
                audio_config=audio_config,
                language=sessions[session_id]["language"]
            )
            
            # Dictionary to store speaker-specific transcriptions
            speaker_transcriptions = {}
            done = False
            
            def handle_result(evt):
                nonlocal done
                if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                    speaker_id = f"speaker_{len(speaker_transcriptions) + 1}"
                    if speaker_id not in speaker_transcriptions:
                        speaker_transcriptions[speaker_id] = []
                    speaker_transcriptions[speaker_id].append(evt.result.text)
                    print(f"Speaker {speaker_id}: {evt.result.text}")
                elif evt.result.reason == speechsdk.ResultReason.NoMatch:
                    print(f"No speech could be recognized: {evt.result.no_match_details}")
                elif evt.result.reason == speechsdk.ResultReason.Canceled:
                    cancellation_details = evt.result.cancellation_details
                    print(f"Speech Recognition canceled: {cancellation_details.reason}")
                    if cancellation_details.reason == speechsdk.CancellationReason.Error:
                        print(f"Error details: {cancellation_details.error_details}")
                done = True
            
            # Subscribe to events
            recognizer.recognized.connect(handle_result)
            recognizer.canceled.connect(handle_result)
            
            # Start recognition
            print("Starting continuous recognition...")
            recognizer.start_continuous_recognition()
            
            # Wait for recognition to complete
            timeout = 10  # 10 seconds timeout
            start_time = time.time()
            
            while not done and (time.time() - start_time) < timeout:
                time.sleep(0.5)
            
            # Ensure recognition is properly stopped
            print("Stopping recognition...")
            recognizer.stop_continuous_recognition()
            recognizer = None  # Release the recognizer
            
            # Format results and identify food ordering speaker
            diarized_text = []
            food_ordering_speaker = None
            
            # Analyze each speaker's text for food ordering context
            for speaker_id, texts in speaker_transcriptions.items():
                combined_text = " ".join(texts)
                
                # Use Groq to analyze if this speaker is ordering food
                prompt = f"""Analyze this text and determine if the speaker is ordering food. 
                Return a JSON with two fields:
                - is_ordering_food (boolean): true if the speaker is clearly ordering food
                - order_details (string): if is_ordering_food is true, extract the order details
                
                Text to analyze: {combined_text}"""
                
                try:
                    completion = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model="mixtral-8x7b-32768",
                        temperature=0.1,
                    )
                    analysis = json.loads(completion.choices[0].message.content)
                    
                    speaker_entry = {
                        "speaker_id": speaker_id,
                        "text": combined_text,
                        "is_ordering_food": analysis.get("is_ordering_food", False),
                        "order_details": analysis.get("order_details", "")
                    }
                    
                                       
                    # diarized_text.append(speaker_entry)
                    print(f"Speaker {speaker_id}: {speaker_entry}")
                    
                    if analysis.get("is_ordering_food", False):
                        food_ordering_speaker = speaker_entry
                        break
                    
                except Exception as e:
                    print(f"Error analyzing speaker {speaker_id}: {str(e)}")
                    # diarized_text.append({
                    #     "speaker_id": speaker_id,
                    #     "text": combined_text,
                    #     "is_ordering_food": False,
                    #     "order_details": ""
                    # })
            
            # send transcription with speaker information and food ordering context
            ws = sessions[session_id].get("websocket")
            if ws:
                message = {
                    "event": "recognized",
                    "diarized_text": diarized_text,
                    "food_ordering_speaker": food_ordering_speaker,
                    "language": sessions[session_id]["language"]
                }
                ws.send(json.dumps(message))
                
        except Exception as e:
            print(f"Error during speech recognition: {str(e)}")
            return jsonify({"error": f"Speech recognition error: {str(e)}"}), 500
            
        finally:
            # Clean up the temporary file
            try:
                # Give some time for the file handle to be released
                time.sleep(1)
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                print(f"Error removing temporary file: {e}")
        
        # Remove from session store
        sessions.pop(session_id, None)
        
        return jsonify({"status": "session_closed", "diarized_text": diarized_text})
    
@sock.route("/ws/chats/<chat_session_id>/sessions/<session_id>")
def speech_socket(ws, chat_session_id, session_id):
    """
    WebSocket endpoint for clients to receive STT results.

    This WebSocket allows clients to connect and receive speech-to-text (STT) results
    in real time. The connection is maintained until the client disconnects. If the 
    session ID is invalid, an error message is sent, and the connection is closed.

    ---
    tags:
      - Sessions
    parameters:
      - name: chat_session_id
        in: path
        type: string
        required: true
        description: The unique identifier for the chat session.
      - name: session_id
        in: path
        type: string
        required: true
        description: The unique identifier for the speech session.
    responses:
      400:
        description: Session not found.
      101:
        description: WebSocket connection established.
    """
    if session_id not in sessions:
        ws.send(json.dumps({"error": "Session not found"}))
        return

    # Store the websocket reference in the session
    sessions[session_id]["websocket"] = ws

    # Keep the socket open to send events
    # Typically we'd read messages from the client in a loop if needed
    while True:
        # If the client closes the socket, an exception is thrown or `ws.receive()` returns None
        msg = ws.receive()
        if msg is None:
            break
              
@app.route('/chats/<chat_session_id>/set-memories', methods=['POST'])
def set_memories(chat_session_id):
    """
    Set memories for a specific chat session.

    ---
    tags:
      - Memories
    parameters:
      - name: chat_session_id
        in: path
        type: string
        required: true
        description: The unique identifier of the chat session.
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            chat_history:
              type: array
              items:
                type: object
                properties:
                  text:
                    type: string
                    description: The chat message text.
              description: List of chat messages in the session.
    responses:
      200:
        description: Memory set successfully.
        schema:
          type: object
          properties:
            success:
              type: string
              example: "1"
      400:
        description: Invalid request data.
    """
    chat_history = request.get_json()
    
    # TODO preprocess data (chat history & system message)
    
    print(f"{chat_session_id} extracting memories for conversation a:{chat_history[-1]['text']}")

    return jsonify({"success": "1"})


@app.route('/chats/<chat_session_id>/get-memories', methods=['GET'])
def get_memories(chat_session_id):
    """
    Retrieve stored memories for a specific chat session.
    ---
    tags:
      - Memories
    parameters:
      - name: chat_session_id
        in: path
        type: string
        required: true
        description: The unique identifier of the chat session.
    responses:
      200:
        description: Successfully retrieved memories for the chat session.
        schema:
          type: object
          properties:
            memories:
              type: string
              description: The stored memories for the chat session.
      400:
        description: Invalid chat session ID.
      404:
        description: Chat session not found.
    """
    print(f"{chat_session_id}: replacing memories...")

    # TODO load relevant memories from your database. Example return value:
    return jsonify({"memories": "The guest typically orders menu 1 and a glass of sparkling water."})


if __name__ == "__main__":
    # In production, you would use a real WSGI server like gunicorn/uwsgi
    app.run(debug=True, host="0.0.0.0", port=8098)