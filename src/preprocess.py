"""
Audio preprocessing functions for noise reduction and speech enhancement.
"""

import numpy as np

def simple_vad(audio_chunk, threshold=0.015, frame_duration=0.02, buffer_duration=1.0):
    """
    Simple voice activity detection based on energy threshold.
    Returns True and the audio chunk if speech is detected, False and None otherwise.
    
    Parameters:
    - audio_chunk: The audio data as a numpy array
    - threshold: Energy threshold for speech detection
    - frame_duration: Duration of each frame in seconds
    - buffer_duration: Duration of buffer to add before and after speech in seconds
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
        # Even if no speech is detected, we'll return the audio chunk anyway
        # This ensures we don't lose any parts of the recording
        print(f"No strong speech detected but keeping audio: {len(audio_chunk)} bytes")
        return True, audio_chunk

def automatic_gain_control(audio_chunk, target_level=-18, time_constant=0.3):
    """
    Automatic gain control to normalize audio volume.
    
    Parameters:
    - audio_chunk: The audio data as a numpy array
    - target_level: Target RMS level in dB (lower = quieter)
    - time_constant: Smoothing factor for gain changes (higher = slower adaptation)
    
    Returns:
    - Volume-normalized audio chunk
    """
    if len(audio_chunk) == 0:
        return audio_chunk
    
    # Initialize static variable for gain memory if not exists
    if not hasattr(automatic_gain_control, "prev_gain"):
        automatic_gain_control.prev_gain = 1.0
        
    # Convert to decibels
    current_level = 20 * np.log10(np.maximum(np.sqrt(np.mean(audio_chunk**2)), 1e-10))
    
    # Calculate gain needed
    gain_db = target_level - current_level
    
    # Limit maximum gain to prevent noise amplification
    gain_db = np.clip(gain_db, -15, 15)  # Limit both amplification and attenuation
    
    # Convert back to linear gain
    gain = 10 ** (gain_db / 20)
    
    # Smooth gain changes using the time constant
    smoothed_gain = time_constant * automatic_gain_control.prev_gain + (1 - time_constant) * gain
    automatic_gain_control.prev_gain = smoothed_gain
    
    # Apply gain
    return audio_chunk * smoothed_gain

def adaptive_noise_reduction(audio_chunk, alpha=0.95, noise_profile=None):
    """
    Adaptive noise reduction using spectral subtraction.
    
    Parameters:
    - audio_chunk: The audio data as a numpy array
    - alpha: Smoothing factor for noise profile update (higher = slower adaptation)
    - noise_profile: Previous noise profile, if any
    
    Returns:
    - Enhanced audio chunk with reduced noise
    """
    if len(audio_chunk) < 10:  # Skip if chunk is too small
        return audio_chunk
        
    # FFT to frequency domain
    spec = np.fft.rfft(audio_chunk)
    mag = np.abs(spec)
    phase = np.angle(spec)
    
    # Initialize or update noise profile
    if noise_profile is None:
        # For first chunk, use a conservative estimate
        adaptive_noise_reduction.noise_profile = mag * 0.5
        print("Initializing noise profile")
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
        # Only update the noise profile with the minimum values to avoid capturing speech
        min_mag = np.minimum(mag, adaptive_noise_reduction.noise_profile)
        adaptive_noise_reduction.noise_profile = alpha * adaptive_noise_reduction.noise_profile + (1 - alpha) * min_mag
    
    # Perform spectral subtraction with a softer reduction factor
    # Use a frequency-dependent reduction factor (more reduction for low frequencies)
    reduction_factor = np.linspace(0.7, 0.3, len(mag))  # Higher reduction for low frequencies
    mag_subtracted = np.maximum(mag - adaptive_noise_reduction.noise_profile * reduction_factor, 0.05 * mag)
    
    # Apply a soft gain function to avoid musical noise
    gain = mag_subtracted / (mag + 1e-10)  # Avoid division by zero
    gain = np.minimum(gain, 1.0)  # Cap the gain at 1.0
    
    # Apply smoothing to the gain to reduce artifacts
    gain_smoothed = np.convolve(gain, np.ones(5)/5, mode='same')
    
    # Apply the smoothed gain
    mag_enhanced = mag * gain_smoothed
    
    # Reconstruct signal
    enhanced = np.fft.irfft(mag_enhanced * np.exp(1j * phase))
    
    # Ensure output length matches input length
    if len(enhanced) > len(audio_chunk):
        enhanced = enhanced[:len(audio_chunk)]
    elif len(enhanced) < len(audio_chunk):
        enhanced = np.pad(enhanced, (0, len(audio_chunk) - len(enhanced)))
    
    return enhanced

def spectral_enhancement(audio_chunk, sampling_rate=16000):
    """
    Enhance speech frequencies while attenuating others.
    
    Parameters:
    - audio_chunk: The audio data as a numpy array
    - sampling_rate: Sampling rate of the audio in Hz
    
    Returns:
    - Enhanced audio with speech frequencies emphasized
    """
    if len(audio_chunk) < 10:  # Skip if chunk is too small
        return audio_chunk
        
    # Define speech-relevant frequency bands (300-3400 Hz for typical speech)
    # We'll use a more detailed frequency response for better speech enhancement
    low_freq = 200
    high_freq = 4000
    
    # Convert to frequency domain
    spec = np.fft.rfft(audio_chunk)
    freq = np.fft.rfftfreq(len(audio_chunk), 1/sampling_rate)
    
    # Create a more detailed filter shape that emphasizes speech frequencies
    # Using a formant-based approach (speech has prominent formants around 500Hz, 1500Hz, 2500Hz)
    gain = np.ones_like(freq)
    
    # Attenuate very low and very high frequencies
    gain[freq < low_freq] = np.linspace(0.05, 0.5, np.sum(freq < low_freq))  # Gradual rolloff for low frequencies
    gain[freq > high_freq] = np.linspace(0.5, 0.05, np.sum(freq > high_freq))  # Gradual rolloff for high frequencies
    
    # Create formant emphasis regions
    # First formant (around 500Hz)
    formant1_mask = (freq >= 300) & (freq <= 800)
    # Second formant (around 1500Hz)
    formant2_mask = (freq >= 1000) & (freq <= 2000)
    # Third formant (around 2500Hz)
    formant3_mask = (freq >= 2200) & (freq <= 3000)
    
    # Boost formant regions
    gain[formant1_mask] = 1.2  # First formant
    gain[formant2_mask] = 1.4  # Second formant (stronger boost)
    gain[formant3_mask] = 1.1  # Third formant (lighter boost)
    
    # Smooth the gain curve to avoid sudden transitions
    win_size = min(31, len(gain) // 10 * 2 + 1)  # Ensure odd window size
    gain = np.convolve(gain, np.hamming(win_size)/np.sum(np.hamming(win_size)), mode='same')
    
    # Apply filter
    enhanced_spec = spec * gain
    
    # Convert back to time domain
    enhanced = np.fft.irfft(enhanced_spec)
    
    # Ensure output length matches input length
    if len(enhanced) > len(audio_chunk):
        enhanced = enhanced[:len(audio_chunk)]
    elif len(enhanced) < len(audio_chunk):
        enhanced = np.pad(enhanced, (0, len(audio_chunk) - len(enhanced)))
    
    # Mix with original to preserve some natural characteristics
    enhanced = 0.7 * enhanced + 0.3 * audio_chunk
    
    return enhanced
