import uuid
import json
import os

from flask import Flask, request, jsonify, send_from_directory
import azure.cognitiveservices.speech as speechsdk
from flask_sock import Sock
from flask_cors import CORS
from flasgger import Swagger

from groq import Groq
from dotenv import load_dotenv
import io

import wave
import numpy as np
from datetime import datetime
import groq
from pathlib import Path

# Import preprocessing functions
from audio_processing.preprocess import (
    simple_vad, 
    automatic_gain_control, 
    adaptive_noise_reduction, 
    spectral_enhancement
)

from memory_module.summarize import summarize_conversation
from memory_module.db import get_customer_profile, update_customer_data
from memory_module.recommender import recommend

load_dotenv()
AZURE_SPEECH_KEY=os.environ.get("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = "switzerlandnorth"
client = groq.Groq(api_key=os.environ.get("GROQ_API_KEY"))

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

last_message = ""

# Create a directory to store processed audio samples
SAMPLES_DIR = Path("processed_samples")
SAMPLES_DIR.mkdir(exist_ok=True)

def ensure_session_fields(session_data):
    """
    Ensure that a session has all the required fields.
    If any field is missing, it will be added with a default value.
    """
    required_fields = {
        "audio_buffer": None,
        "original_audio_path": None,
        "processed_audio_path": None,
        "websocket": None
    }
    
    for field, default_value in required_fields.items():
        if field not in session_data:
            session_data[field] = default_value
    
    return session_data

def transcribe_whisper(audio_recording):
    try:
        # Check if input is a file path or raw audio data
        if isinstance(audio_recording, str) and os.path.exists(audio_recording):
            # Input is a file path
            with open(audio_recording, 'rb') as f:
                audio_file = io.BytesIO(f.read())
                audio_file.name = os.path.basename(audio_recording)
        else:
            # Input is raw audio data
            audio_file = io.BytesIO(audio_recording)
            audio_file.name = 'audio.wav'  # Whisper requires a filename with a valid extension
            
        transcription = client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=audio_file,
            #language = ""  # specify Language explicitly
        )
        # print(f"openai transcription: {transcription.text}")
        return transcription.text
    except Exception as e:
        print(f"Error during Groq API call: {str(e)}")
        return "Transcription failed due to connection error"

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
        "websocket": None,  # will be set when the client connects via WS (WebSocket)
        "original_audio_path": None,
        "processed_audio_path": None
    }
    
    print(f"DEBUG - Created session: {session_id}")
    print(f"DEBUG - Session object: {sessions[session_id]}")

    return jsonify({"session_id": session_id})

@app.route("/chats/<chat_session_id>/sessions/<session_id>/wav", methods=["POST"])
def upload_audio_chunk(chat_session_id, session_id):
    """
    Upload an audio chunk (expected 16kb, ~0.5s of WAV data).
    The chunk is appended to the push stream for the session.
    ---
    tags:
      - Audio
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
        description: The unique identifier of the session.
      - name: audio_chunk
        in: body
        required: true
        description: The audio chunk data to upload.
    responses:
      200:
        description: Audio chunk uploaded successfully.
        schema:
          type: object
          properties:
            original_audio:
              type: string
              description: Path to the original audio file.
            processed_audio:
              type: string
              description: Path to the processed audio file.
      400:
        description: Invalid request data.
    """
    # Check if session exists
    if session_id not in sessions:
        # Initialize session if it doesn't exist
        print(f"Initializing new session: {session_id}")
        sessions[session_id] = {}
    
    # Ensure session has all required fields
    sessions[session_id] = ensure_session_fields(sessions[session_id])
    
    # Get audio data from request
    audio_data = request.data
    print(f"Received audio chunk: {len(audio_data)} bytes")
    
    if len(audio_data) == 0:
        return jsonify({"error": "Empty audio data"}), 400
    
    # Check if this is the first chunk
    is_first_chunk = not sessions[session_id].get("original_audio_path")
    
    # Save original audio to file
    if not sessions[session_id].get("original_audio_path"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = f"{SAMPLES_DIR}/original_{session_id}_{timestamp}.wav"
        sessions[session_id]["original_audio_path"] = original_filename
        
        # Initialize the file with WAV header
        with wave.open(original_filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(16000)  # Assuming 16kHz sampling rate
            wf.writeframes(audio_data)
    else:
        # Append to existing file
        original_audio_path = sessions[session_id].get("original_audio_path")
        existing_audio = b''
        params = None
        
        try:
            with wave.open(original_audio_path, 'rb') as wf:
                params = wf.getparams()
                existing_audio = wf.readframes(wf.getnframes())
                print(f"Original audio: existing frames: {len(existing_audio)} bytes")
        except (FileNotFoundError, wave.Error):
            # If file doesn't exist or is empty, create a new one
            params = (1, 2, 16000, 0, 'NONE', 'not compressed')
            print("Creating new original audio file")
        
        print(f"New audio chunk size: {len(audio_data)} bytes")
        
        # Write combined audio data
        with wave.open(original_audio_path, 'wb') as wf:
            wf.setparams(params)
            combined_audio = existing_audio + audio_data
            print(f"Combined audio size: {len(combined_audio)} bytes")
            wf.writeframes(combined_audio)
    
    # Convert bytes to numpy array
    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    
    # Normalize to float
    audio_normalized = audio_array.astype(np.float32) / 32768.0
    
    # Initialize processing functions if first chunk
    if not hasattr(upload_audio_chunk, "noise_profile"):
        upload_audio_chunk.noise_profile = None
        upload_audio_chunk.prev_chunk = None
    
    # If we have a previous chunk stored, prepend it to create overlap for smoother processing
    if upload_audio_chunk.prev_chunk is not None and len(upload_audio_chunk.prev_chunk) > 0:
        # Use overlap of 25% of the previous chunk for smooth transitions
        overlap_size = len(upload_audio_chunk.prev_chunk) // 4
        if overlap_size > 0:
            overlap = upload_audio_chunk.prev_chunk[-overlap_size:]
            audio_normalized = np.concatenate([overlap, audio_normalized])
            print(f"Added {overlap_size} samples of overlap from previous chunk")
    
    # Apply preprocessing pipeline
    has_speech, processed_audio = simple_vad(audio_normalized, threshold=0.015)
    
    # Always store the original audio data in the buffer
    if sessions[session_id]["audio_buffer"] is not None:
        print(f"Existing audio buffer size: {len(sessions[session_id]['audio_buffer'])} bytes")
        sessions[session_id]["audio_buffer"] = sessions[session_id]["audio_buffer"] + audio_data
        print(f"Updated audio buffer size (original): {len(sessions[session_id]['audio_buffer'])} bytes")
    else:
        print(f"Initializing audio buffer with original audio: {len(audio_data)} bytes")
        sessions[session_id]["audio_buffer"] = audio_data
    
    # Process audio for noise reduction
    if processed_audio is not None:
        try:
            # Keep track of if this is the first or last chunk
            is_first_chunk = not sessions[session_id].get("processed_audio_path")
            
            processed_audio = automatic_gain_control(processed_audio)
            processed_audio = adaptive_noise_reduction(processed_audio, 
                                                   noise_profile=upload_audio_chunk.noise_profile)
            upload_audio_chunk.noise_profile = adaptive_noise_reduction.noise_profile
            processed_audio = spectral_enhancement(processed_audio)
            
            # No silence buffer added to regular chunks to avoid choppiness
        except Exception as e:
            print(f"Error during audio processing: {str(e)}")
            # Fall back to original audio if processing fails
            processed_audio = audio_normalized
    else:
        processed_audio = audio_normalized
    
    # Store the end of this chunk for use with the next chunk (for smooth transitions)
    store_size = min(4000, len(audio_normalized))  # Store up to 250ms at 16kHz
    upload_audio_chunk.prev_chunk = audio_normalized[-store_size:].copy()
    
    # If we added overlap from the previous chunk, remove it from the processed audio
    if upload_audio_chunk.prev_chunk is not None and 'overlap_size' in locals() and overlap_size > 0:
        processed_audio = processed_audio[overlap_size:]
    
    # Convert back to int16 for storage
    processed_int16 = (processed_audio * 32768).astype(np.int16)
    processed_bytes = processed_int16.tobytes()
    
    # Always save processed audio to file, regardless of speech detection
    if not sessions[session_id].get("processed_audio_path"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        processed_filename = f"{SAMPLES_DIR}/processed_{session_id}_{timestamp}.wav"
        sessions[session_id]["processed_audio_path"] = processed_filename
        
        # Initialize the processed audio file with WAV header
        with wave.open(processed_filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(16000)
            wf.writeframes(processed_bytes)
    else:
        # Append processed audio to the file
        processed_audio_path = sessions[session_id].get("processed_audio_path")
        if processed_audio_path:
            # Read existing audio data
            existing_audio = b''
            try:
                with wave.open(processed_audio_path, 'rb') as wf:
                    params = wf.getparams()
                    existing_audio = wf.readframes(wf.getnframes())
                    print(f"Processed audio: existing frames: {len(existing_audio)} bytes")
            except (FileNotFoundError, wave.Error):
                # If file doesn't exist or is empty, create a new one
                params = (1, 2, 16000, 0, 'NONE', 'not compressed')
                print("Creating new processed audio file")
            
            print(f"New processed audio chunk size: {len(processed_bytes)} bytes")
            
            # Write combined audio data
            with wave.open(processed_audio_path, 'wb') as wf:
                wf.setparams(params)
                combined_audio = existing_audio + processed_bytes
                print(f"Combined processed audio size: {len(combined_audio)} bytes")
                wf.writeframes(combined_audio)
    
    # Get file paths for response
    original_audio_path = sessions[session_id].get("original_audio_path")
    processed_audio_path = sessions[session_id].get("processed_audio_path")
    
    original_audio = os.path.basename(original_audio_path) if original_audio_path else ""
    processed_audio = os.path.basename(processed_audio_path) if processed_audio_path else ""
    
    # Return the paths to audio files along with status
    return jsonify({
        "status": "audio_chunk_received",
        "original_audio": original_audio,
        "processed_audio": processed_audio
    })

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

    # Ensure session has all required fields
    sessions[session_id] = ensure_session_fields(sessions[session_id])
        
    # Process final audio buffer
    if sessions[session_id]["audio_buffer"] is not None:
        print(f"Final audio buffer size: {len(sessions[session_id]['audio_buffer'])} bytes")
        
        # Add a tail buffer to the final processed audio file to prevent voice cutoff
        processed_audio_path = sessions[session_id].get("processed_audio_path")
        if processed_audio_path:
            try:
                # Read the processed audio file
                with wave.open(processed_audio_path, 'rb') as wf:
                    params = wf.getparams()
                    existing_audio = wf.readframes(wf.getnframes())
                
                # Add a 0.5 second silence buffer at the end
                tail_buffer_size = int(0.5 * 16000)  # 500ms at 16kHz
                tail_buffer = np.zeros(tail_buffer_size, dtype=np.int16).tobytes()
                
                # Write combined audio with tail buffer
                with wave.open(processed_audio_path, 'wb') as wf:
                    wf.setparams(params)
                    wf.writeframes(existing_audio + tail_buffer)
                    print(f"Added final tail buffer of {tail_buffer_size} samples to processed audio")
            except Exception as e:
                print(f"Error adding tail buffer: {str(e)}")
        
        try:
            # Use processed audio file if available, otherwise fall back to audio buffer
            processed_audio_path = sessions[session_id].get("processed_audio_path")

            # Use pick_relevant_speaker from diarizer.py to get the speaker who is talking to us
            from diarizer import pick_relevant_speaker

            if processed_audio_path and os.path.exists(processed_audio_path):
                print(f"Using processed audio file for speaker diarization: {processed_audio_path}")
                # Call pick_relevant_speaker with the processed audio path and session_id
                relevant_speaker = pick_relevant_speaker(processed_audio_path, session_id)
                
                # Store the relevant speaker information in the session
                if relevant_speaker:
                    sessions[session_id]["relevant_speaker"] = relevant_speaker
                    print(f"Relevant speaker identified and stored in session")
                
                # Get the websocket to send the transcription back to the client
                ws = sessions[session_id].get("websocket")
                if ws and relevant_speaker:
                    # The pick_relevant_speaker function already sends the message via websocket
                    # We don't need to send it again here
                    print(f"Relevant speaker identified and message sent via websocket")
            else:
                print("Processed audio file not available, cannot perform speaker diarization")
        except Exception as e:
            print(f"Error during speaker diarization: {str(e)}")

    
    # Get file paths before removing session
    original_audio_path = sessions[session_id].get("original_audio_path")
    processed_audio_path = sessions[session_id].get("processed_audio_path")
    
    # Store the relevant speaker information if it was found
    relevant_speaker_info = sessions[session_id].get("relevant_speaker")
    
    # Handle None values
    if original_audio_path is None:
        original_audio = ""
    else:
        original_audio = os.path.basename(original_audio_path)
        
    if processed_audio_path is None:
        processed_audio = ""
    else:
        processed_audio = os.path.basename(processed_audio_path)
    
    # Remove from session store
    sessions.pop(session_id, None)

    response = {
        "status": "session_closed",
        "original_audio": original_audio,
        "processed_audio": processed_audio,
        "message": "Audio files can be accessed at /samples/{filename}"
    }
    
    # Add relevant speaker information if available
    if relevant_speaker_info:
        response["relevant_speaker"] = relevant_speaker_info
        
    return jsonify(response)

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

    # Ensure session has all required fields
    sessions[session_id] = ensure_session_fields(sessions[session_id])
    
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
    speaker_chats = [item for item in chat_history if item['type'] == 0]
    last_message = speaker_chats[-1]['text']
    print(f"[SET MEMORIES] Last message is: {last_message}")
    customer_data = get_customer_profile('Ahmed')
    new_data = summarize_conversation(last_message, customer_data)
    update_customer_data('Ahmed', new_data)

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
    return jsonify({"memories":f"{recommend(last_input=last_message, customer_data=get_customer_profile('Ahmed'))}"})

# Add an endpoint to retrieve the audio files
@app.route("/samples/<filename>", methods=["GET"])
def get_audio_sample(filename):
    """
    Retrieve a saved audio sample file.
    ---
    tags:
      - Samples
    parameters:
      - name: filename
        in: path
        type: string
        required: true
        description: Name of the audio sample file
    responses:
      200:
        description: Audio file
        content:
          audio/wav:
            schema:
              type: string
              format: binary
      404:
        description: File not found
        schema:
          type: object
          properties:
            error:
              type: string
              description: Error message
    """
    file_path = SAMPLES_DIR / filename
    if not file_path.exists():
        return jsonify({"error": "File not found"}), 404
    
    return send_from_directory(str(SAMPLES_DIR), filename, mimetype="audio/wav")

if __name__ == "__main__":
    # In production, you would use a real WSGI server like gunicorn/uwsgi
    app.run(debug=True, host="0.0.0.0", port=5000)