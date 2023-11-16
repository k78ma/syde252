import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
from scipy import signal

def convert_m4a_to_wav(m4a_path, wav_path):
    audio = AudioSegment.from_file(m4a_path, format="m4a")
    audio.export(wav_path, format="wav")

def convert_mp3_to_wav(mp3_path, wav_path):
    audio = AudioSegment.from_file(mp3_path, format="mp3")
    audio.export(wav_path, format="wav")

def play_sound(data, samplerate):
    # Ensure the data is in the range [-1, 1] and of type float32
    if data.dtype.kind == 'i':  # Check if the data is of integer type
        data_normalized = data.astype(np.float32) / np.iinfo(data.dtype).max
    else:  # If it's already a float, do not normalize again
        data_normalized = data.astype(np.float32)
        
    sd.play(data_normalized, samplerate)
    sd.wait()


def get_audio_info_from_mp3(file_path):
    audio = AudioSegment.from_mp3(file_path)
    data = np.array(audio.get_array_of_samples())
    
    # If stereo, take the average of both channels to make it mono
    if audio.channels == 2:
        data = data.reshape((-1, 2))
        data = np.mean(data, axis=1)
        print("Converted to Mono")
    
    return data, audio.frame_rate

def write_sound(file_path, data, samplerate):
    sf.write(file_path, data, samplerate)

def downsample_audio(data, original_sr, target_sr):
    # Calculate the target number of samples
    num_samples = int(len(data) * (target_sr / original_sr))
    
    # Use scipy's resample function
    resampled_data = signal.resample(data, num_samples)

    print("Downsampled from", original_sr, "Hz to", target_sr, "Hz")
    
    return resampled_data

def plot_waveform_time(data, samplerate):
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(data.shape[0])/samplerate, data)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Waveform")
    plt.show()

def plot_waveform_samplenumber(data):
    plt.figure(figsize=(10, 4))
    plt.plot(data)
    plt.title("Waveform")
    plt.xlabel("Sample Number")
    plt.ylabel("Amplitude")
    plt.show()

def process_and_analyze_audio(data, samplerate, new_file_path):
    # Get audio info and convert to mono if stereo
    if samplerate != 16000:
        data = downsample_audio(data, samplerate, 16000)
        samplerate = 16000
    
    # Play the sound
    #play_sound(data, samplerate)
    
    # Write the sound to a new file
    write_sound(new_file_path, data, samplerate)
    
    # Plot the waveform
    plot_waveform_samplenumber(data)

def generate_and_play_cosine_wave(input_signal, sampling_rate):
    """
    Generate a 1 kHz cosine wave, play it, and plot its first two cycles.
    
    Parameters:
        input_signal (numpy array): The input signal array.
        sampling_rate (int): The sampling rate of the input signal.
    """
    print("Generating and playing a 1 kHz cosine wave...")
    # 1. Generate a 1 kHz cosine wave
    frequency = 1000  # Frequency of the cosine wave in Hz
    time_array = np.arange(len(input_signal)) / sampling_rate
    cosine_wave = np.cos(2 * np.pi * frequency * time_array)

    # 2. Ensure the cosine wave has the same duration as the input signal
    # (this should be automatically ensured by using the same `time_array` length)

    # 3. Play the sound
    cosine_wave_normalized = cosine_wave / np.max(np.abs(cosine_wave))
    sd.play(cosine_wave_normalized, samplerate=sampling_rate)
    sd.wait()  # Wait until audio playback is finished

    # 4. Plot two cycles of the waveform as a function of time
    num_samples_two_cycles = int((2/frequency) * sampling_rate)
    plt.plot(time_array[:num_samples_two_cycles], cosine_wave[:num_samples_two_cycles])
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("1 kHz Cosine Wave (Two Cycles)")
    plt.grid(True)
    plt.show()


def get_audio_info(file_path):
    data, samplerate = sf.read(file_path)
    print(f"Sampling rate: {samplerate} Hz")
    
    channels = data.shape[1] if len(data.shape) > 1 else 1
    print("Channels:", "Stereo" if channels == 2 else "Mono")
    
    if channels == 2:
        data = np.mean(data, axis=1)  # summing channels may exceed max amplitude, so mean is safer
        print("Converted to Mono")
    
    duration_seconds = len(data) / samplerate
    print(f"Duration: {duration_seconds:.2f} seconds")
        
    return data, samplerate

if __name__ == "__main__":
    # Example usage:
    new_file_path = '/Users/kai/code/2b/syde252/downsampled_bohemian.wav'

    convert_mp3_to_wav('/Users/kai/code/2b/syde252/Bohemian Rhapsody.mp3', './Bohemian Rhapsody.wav')

    data, samplerate = get_audio_info('Bohemian Rhapsody.wav')

    # Process and analyze audio
    process_and_analyze_audio(data, samplerate, new_file_path)
    generate_and_play_cosine_wave(data, samplerate)
