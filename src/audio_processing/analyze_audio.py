import wave
import numpy as np
import os
import sys

def analyze_wav_file(file_path):
    """Analyze a WAV file and print its properties."""
    try:
        with wave.open(file_path, 'rb') as wf:
            # Get basic properties
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frame_rate = wf.getframerate()
            n_frames = wf.getnframes()
            duration = n_frames / frame_rate
            
            # Read all frames
            frames = wf.readframes(n_frames)
            
            # Convert to numpy array for analysis
            if sample_width == 2:  # 16-bit audio
                dtype = np.int16
            elif sample_width == 1:  # 8-bit audio
                dtype = np.uint8
            else:
                dtype = np.int32
                
            samples = np.frombuffer(frames, dtype=dtype)
            
            # Calculate statistics
            if len(samples) > 0:
                max_amplitude = np.max(np.abs(samples))
                mean_amplitude = np.mean(np.abs(samples))
                rms = np.sqrt(np.mean(samples.astype(np.float32)**2))
                
                # Count non-zero samples (potential speech)
                non_zero_threshold = 100  # Threshold to consider as non-zero
                non_zero_samples = np.sum(np.abs(samples) > non_zero_threshold)
                non_zero_percentage = (non_zero_samples / len(samples)) * 100
                
                # Detect potential silence gaps
                frame_size = int(frame_rate * 0.02)  # 20ms frames
                if len(samples) >= frame_size:
                    frames = [samples[i:i+frame_size] for i in range(0, len(samples), frame_size) if i+frame_size <= len(samples)]
                    frame_energies = [np.sum(frame**2) for frame in frames]
                    silent_frames = sum(1 for energy in frame_energies if energy < 1000)
                    silent_percentage = (silent_frames / len(frames)) * 100
                else:
                    silent_percentage = 0
            else:
                max_amplitude = 0
                mean_amplitude = 0
                rms = 0
                non_zero_percentage = 0
                silent_percentage = 0
            
            print(f"File: {os.path.basename(file_path)}")
            print(f"  Channels: {channels}")
            print(f"  Sample Width: {sample_width} bytes")
            print(f"  Frame Rate: {frame_rate} Hz")
            print(f"  Number of Frames: {n_frames}")
            print(f"  Duration: {duration:.2f} seconds")
            print(f"  File Size: {os.path.getsize(file_path)} bytes")
            print(f"  Max Amplitude: {max_amplitude}")
            print(f"  Mean Amplitude: {mean_amplitude:.2f}")
            print(f"  RMS: {rms:.2f}")
            print(f"  Non-zero Content: {non_zero_percentage:.2f}%")
            print(f"  Silent Content: {silent_percentage:.2f}%")
            print()
            
    except Exception as e:
        print(f"Error analyzing {file_path}: {str(e)}")

def main():
    # Get the directory containing audio files
    samples_dir = "processed_samples"
    
    # Find all WAV files in the directory
    wav_files = [os.path.join(samples_dir, f) for f in os.listdir(samples_dir) if f.endswith('.wav')]
    
    # Analyze each file
    for wav_file in wav_files:
        analyze_wav_file(wav_file)

if __name__ == "__main__":
    main()
