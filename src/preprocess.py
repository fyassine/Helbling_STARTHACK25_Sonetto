"""
Audio preprocessing functions for noise reduction and speech enhancement.
"""

import numpy as np

def simple_vad(audio_chunk, threshold=0.015, frame_duration=0.02):
    """
    Simple voice activity detection based on energy threshold.
    Returns True and the audio chunk if speech is detected, False and None otherwise.
    """
    if len(audio_chunk) < 10:  # Skip if chunk is too small
        return False, None
        
    # Calculate short-time energy
    frame_length = max(int(frame_duration * 16000), 1)  # Assuming 16kHz sampling rate
    frames = []
    for i in range(0, len(audio_chunk) - frame_length, frame_length):
        frames.append(audio_chunk[i:i+frame_length])
    
    if not frames:  # If no frames could be created
        return False, None
        
    frames = np.array(frames)
    energy = np.sum(frames**2, axis=1) / frame_length
    
    # Apply threshold
    speech_frames = energy > threshold
    
    speech_percentage = np.mean(speech_frames) * 100
    print(f"Speech detection: {speech_percentage:.2f}% of frames have speech (threshold: {threshold})")
    
    # Only process if speech is detected
    if np.mean(speech_frames) > 0.05:  # At least 5% of frames have speech
        print(f"Speech detected: returning {len(audio_chunk)} bytes")
        return True, audio_chunk
    else:
        print(f"No speech detected: discarding {len(audio_chunk)} bytes")
        return False, None

def automatic_gain_control(audio_chunk, target_level=-15, time_constant=0.1):
    """
    Automatic gain control to normalize audio volume.
    """
    if len(audio_chunk) == 0:
        return audio_chunk
        
    # Convert to decibels
    current_level = 20 * np.log10(np.maximum(np.sqrt(np.mean(audio_chunk**2)), 1e-10))
    
    # Calculate gain needed
    gain_db = target_level - current_level
    
    # Limit maximum gain to prevent noise amplification
    gain_db = min(gain_db, 20)
    
    # Convert back to linear gain
    gain = 10 ** (gain_db / 20)
    
    # Apply gain
    return audio_chunk * gain

def adaptive_noise_reduction(audio_chunk, alpha=0.95, noise_profile=None):
    """
    Adaptive noise reduction using spectral subtraction.
    """
    if len(audio_chunk) < 10:  # Skip if chunk is too small
        return audio_chunk
        
    # FFT to frequency domain
    spec = np.fft.rfft(audio_chunk)
    mag = np.abs(spec)
    phase = np.angle(spec)
    
    # Initialize or update noise profile
    if noise_profile is None:
        adaptive_noise_reduction.noise_profile = mag
    else:
        # Make sure the noise profile has the same shape as the current magnitude
        if len(noise_profile) != len(mag):
            print(f"Noise profile shape mismatch: {len(noise_profile)} vs {len(mag)}")
            # Resize noise profile to match current magnitude
            if len(noise_profile) > len(mag):
                noise_profile = noise_profile[:len(mag)]
            else:
                # Pad with zeros
                noise_profile = np.pad(noise_profile, (0, len(mag) - len(noise_profile)))
        
        adaptive_noise_reduction.noise_profile = noise_profile
        # Update noise profile with a smoothing factor
        adaptive_noise_reduction.noise_profile = alpha * adaptive_noise_reduction.noise_profile + (1 - alpha) * mag
    
    # Perform spectral subtraction
    mag_subtracted = np.maximum(mag - adaptive_noise_reduction.noise_profile * 0.5, 0.01 * mag)
    
    # Reconstruct signal
    enhanced = np.fft.irfft(mag_subtracted * np.exp(1j * phase))
    
    # Ensure output length matches input length
    if len(enhanced) > len(audio_chunk):
        enhanced = enhanced[:len(audio_chunk)]
    elif len(enhanced) < len(audio_chunk):
        enhanced = np.pad(enhanced, (0, len(audio_chunk) - len(enhanced)))
    
    return enhanced

def spectral_enhancement(audio_chunk, sampling_rate=16000):
    """
    Enhance speech frequencies while attenuating others.
    """
    if len(audio_chunk) < 10:  # Skip if chunk is too small
        return audio_chunk
        
    # Define speech-relevant frequency bands (300-3400 Hz for typical speech)
    low_freq = 300
    high_freq = 3400
    
    # Convert to frequency domain
    spec = np.fft.rfft(audio_chunk)
    freq = np.fft.rfftfreq(len(audio_chunk), 1/sampling_rate)
    
    # Create a bandpass filter that emphasizes speech frequencies
    gain = np.ones_like(freq)
    gain[freq < low_freq] = 0.1  # Attenuate low frequencies
    gain[freq > high_freq] = 0.1  # Attenuate high frequencies
    
    # Apply gain to middle frequencies (speech range)
    speech_mask = (freq >= low_freq) & (freq <= high_freq)
    gain[speech_mask] = 1.5  # Boost speech frequencies
    
    # Apply filter
    enhanced_spec = spec * gain
    
    # Convert back to time domain
    enhanced = np.fft.irfft(enhanced_spec)
    
    # Ensure output length matches input length
    if len(enhanced) > len(audio_chunk):
        enhanced = enhanced[:len(audio_chunk)]
    elif len(enhanced) < len(audio_chunk):
        enhanced = np.pad(enhanced, (0, len(audio_chunk) - len(enhanced)))
    
    return enhanced
