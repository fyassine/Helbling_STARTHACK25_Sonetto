def process_audio_file(audio_file_path):
    try:
        audio_file = None
        # Check if input is a file path or raw audio data
        if isinstance(audio_recording, str) and os.path.exists(audio_recording):
            # Input is a file path
            with open(audio_recording, 'rb') as f:
                audio_file = io.BytesIO(f.read())
                audio_file.name = os.path.basename(audio_recording)
        else:
            # Input is raw audio data
            audio_file = io.BytesIO(audio_recording)
            audio_file.name = 'audio.wav'

        return audio_file

    except Exception as e:
        print(f"Error during Groq API call: {str(e)}")
        return "Transcription failed due to connection error"

def transcribe_all_speakers(audio_file_path):       
    # transcription = client.audio.transcriptions.create(
    #     model="whisper-large-v3",
    #     file=audio_file,
    #     #language = ""  # specify Language explicitly
    # )
    # print(f"openai transcription: {transcription.text}")
    # return transcription.text

    audio_file = process_audio_file(audio_file_path)

    try:
        # Initialize Azure Speech config
        speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
        print(f"Using Azure Speech Service with region: {AZURE_SPEECH_REGION}")
        
        # Create audio config from the temporary file
        audio_config = speechsdk.audio.AudioConfig(filename=audio_file)
        
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
    
        return speaker_transcriptions

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

def pick_relevant_speaker(audio_file_path):

    speaker_transcriptions = transcribe_all_speakers(audio_file_path)
        
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
            
    # Remove from session store
    sessions.pop(session_id, None)
    
    return jsonify({
        "status": "session_closed",
        "diarized_text": diarized_text,
        "original_audio": original_audio,
        "processed_audio": processed_audio,
        "message": "Audio files can be accessed at /samples/{filename}"
    })