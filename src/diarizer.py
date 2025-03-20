import os
import io
import json
import time
import azure.cognitiveservices.speech as speechsdk
from flask import jsonify

# Import will happen at runtime to avoid circular imports
# from relay import sessions, AZURE_SPEECH_KEY, AZURE_SPEECH_REGION

def process_audio_file(audio_file_path):
    """Process audio file for transcription.
    
    Args:
        audio_file_path: Path to the audio file or raw audio data
        
    Returns:
        Processed audio file object ready for transcription
    """
    
    try:
        audio_file = None
        # Check if input is a file path or raw audio data
        if isinstance(audio_file_path, str) and os.path.exists(audio_file_path):
            # Input is a file path
            with open(audio_file_path, 'rb') as f:
                audio_file = io.BytesIO(f.read())
                audio_file.name = os.path.basename(audio_file_path)
        else:
            # Input is raw audio data
            audio_file = io.BytesIO(audio_file_path)
            audio_file.name = 'audio.wav'

        return audio_file

    except Exception as e:
        print(f"Error processing audio file: {str(e)}")
        return None

def transcribe_all_speakers(audio_file_path):
    """Transcribe audio and identify different speakers.
    
    Args:
        audio_file_path: Path to the audio file to transcribe
        
    Returns:
        Dictionary mapping speaker IDs to lists of transcribed text segments
    """
    # Import the Azure Speech credentials from relay.py at runtime
    # to avoid circular imports
    from relay import AZURE_SPEECH_KEY, AZURE_SPEECH_REGION
    
    try:
        # Initialize Azure Speech config with appropriate settings
        speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
        speech_config.speech_recognition_language = "en-US"  # Set recognition language
        speech_config.request_word_level_timestamps()  # Request word timestamps for better accuracy
        speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "5000")
        speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs, "5000")
        print(f"Using Azure Speech Service with region: {AZURE_SPEECH_REGION}")
        
        # Create audio config from the audio file path
        audio_config = speechsdk.audio.AudioConfig(filename=audio_file_path)
        
        # Create speech recognizer
        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
        
        # Dictionary to store speaker-specific transcriptions
        speaker_transcriptions = {}
        
        # Simple approach: use recognize_once() which is more reliable for short audio clips
        print("Starting speech recognition...")
        result = recognizer.recognize_once()
        
        # Process the result
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            text = result.text
            print(f"Recognized: {text}")
            
            # For simplicity, assign all text to a single speaker
            speaker_id = "speaker_1"
            speaker_transcriptions[speaker_id] = [text]
            
        elif result.reason == speechsdk.ResultReason.NoMatch:
            print(f"No speech could be recognized: {result.no_match_details.reason}")
            if result.no_match_details.reason == speechsdk.NoMatchReason.InitialSilenceTimeout:
                print("The recording started with silence, and the service timed out waiting for speech.")
            elif result.no_match_details.reason == speechsdk.NoMatchReason.InitialBabbleTimeout:
                print("The recording started with noise that wasn't recognized as speech.")
            
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print(f"Speech Recognition canceled: {cancellation_details.reason}")
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print(f"Error details: {cancellation_details.error_details}")
                
        # Alternative approach: try to use the speech-to-text REST API directly if the SDK approach fails
        if not speaker_transcriptions:
            try:
                # Fallback to a simpler approach if the first one failed
                print("Trying alternative recognition approach...")
                audio_input = speechsdk.AudioConfig(filename=audio_file_path)
                speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_input)
                
                # Use a simple synchronous recognition
                simple_result = speech_recognizer.recognize_once_async().get()
                if simple_result.reason == speechsdk.ResultReason.RecognizedSpeech:
                    print(f"Alternative approach recognized: {simple_result.text}")
                    speaker_transcriptions["speaker_1"] = [simple_result.text]
            except Exception as e:
                print(f"Alternative approach failed: {str(e)}")
        
        # If no speakers were detected, add a default one with empty text
        if not speaker_transcriptions:
            speaker_transcriptions["speaker_1"] = [""]
            print("No speech detected, adding default empty speaker")
        
        return speaker_transcriptions

    except Exception as e:
        print(f"Error during speech recognition: {str(e)}")
        # Return an empty dictionary instead of a Flask response
        return {"speaker_1": [f"Error: {str(e)}"]}

def pick_relevant_speaker(audio_file_path, session_id=None):
    """Diarize audio and identify the most relevant speaker.
    
    Args:
        audio_file_path: Path to the audio file to process
        session_id: Optional session ID to retrieve session information
        
    Returns:
        The relevant speaker information or None if no relevant speaker was found
    """
    # Import the sessions dictionary from relay.py at runtime
    # to avoid circular imports
    from relay import sessions
    
    # Get speaker transcriptions using Azure Speech SDK
    speaker_transcriptions = transcribe_all_speakers(audio_file_path)
        
    # Format results and identify the relevant speaker
    diarized_text = []
    relevant_speaker = None

    # Analyze each speaker's text to find the most relevant one
    for speaker_id, texts in speaker_transcriptions.items():
        combined_text = " ".join(texts)
        
        # Create a speaker entry for this speaker
        speaker_entry = {
            "speaker_id": speaker_id,
            "text": combined_text,
        }
        
        # Add this speaker to the diarized text list
        diarized_text.append(speaker_entry)
        print(f"Speaker {speaker_id}: {combined_text}")
        
        # For now, we'll consider the first speaker with non-empty text as relevant
        # This is a simple approach - in a real system, you might want more sophisticated logic
        if not relevant_speaker and combined_text.strip():
            relevant_speaker = speaker_entry
    
    # If no relevant speaker was found but we have speakers, use the first one
    if not relevant_speaker and diarized_text:
        relevant_speaker = diarized_text[0]
        print(f"No speaker with non-empty text found, using first speaker: {relevant_speaker['speaker_id']}")
        
    # If we still don't have a relevant speaker, create a default one
    if not relevant_speaker:
        relevant_speaker = {
            "speaker_id": "default_speaker",
            "text": "No speech detected"
        }
        diarized_text.append(relevant_speaker)
        print("No speakers detected, using default speaker")
    
    # If session_id is provided and valid, send the results via websocket
    if session_id and session_id in sessions:
        ws = sessions[session_id].get("websocket")
        if ws:
            message = {
                "event": "recognized",
                "diarized_text": diarized_text,
                "relevant_speaker": relevant_speaker,
                "language": sessions[session_id]["language"]
            }
            ws.send(json.dumps(message))
            print(f"Sent diarized text with {len(diarized_text)} speakers via websocket")
    
    return relevant_speaker