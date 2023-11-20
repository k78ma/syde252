from phase1 import get_audio_info, process_and_analyze_audio
import numpy as np
from scipy.signal import butter, firwin, lfilter
from scipy.signal import firwin, filtfilt, get_window
import matplotlib.pyplot as plt

#### DEFINE FREQUENCY BANDS FROM 100 Hz to 8 kHz
# Logarithmic scale is used because it represents human hearing
def define_frequency_bands(N, min_freq=100, max_freq=8000):
    # Use logarithmic scale for frequency bands
    log_min = np.log10(min_freq)
    log_max = np.log10(max_freq)
    log_band_edges = np.logspace(log_min, log_max, N+1)

    return [(log_band_edges[i], log_band_edges[i+1]) for i in range(N)]

### Bandpass Filter
### FIR filter is used because it is linear phase and stable but may require higher filter order
### IIR filter are more computationally efficient (lower order for the same performance) but can have nonlinear phase responses, which might affect the signal.
def create_bandpass_filter(frequencies, samplerate, filter_type='IIR', order=4):
    low, high = frequencies

    nyquist = 0.5 * samplerate

    low_normalized = max(min(low / nyquist, 0.99), 0.01) # Avoid boundaries to avoid issues
    high_normalized = max(min(high / nyquist, 0.99), 0.01)

    if filter_type == 'IIR':
        b, a = butter(order, [low_normalized, high_normalized], btype='bandpass')
        return lambda data: lfilter(b, a, data)
    elif filter_type == 'FIR':
        # Ensure the frequencies are within the valid range
        low = max(low, 1)  # Prevent 0 frequency
        high = min(high, samplerate / 2 - 1)  # Prevent going above Nyquist frequency
        b = firwin(order, [low, high], pass_zero=False, fs=samplerate)
        return lambda data: np.convolve(data, b, mode='same')


### Rectify by taking absolute value
def rectify_signal(signal):
    return np.abs(signal)

### Lowpass filter
def create_lowpass_filter(cutoff, samplerate, filter_type='FIR', order=100):
    # Normalized cutoff frequency
    normalized_cutoff = cutoff / (0.5 * samplerate)

    if filter_type == 'FIR':
        # Pass a string or tuple for the window type directly to firwin
        b = firwin(order+1, normalized_cutoff, window='hamming', pass_zero='lowpass', fs=samplerate)
        # Use filtfilt to apply the filter forward and backward
        return lambda data: filtfilt(b, [1], data)
    else:
        # Implementation for IIR filter if needed
        b, a = butter(order, normalized_cutoff, btype='low')
        return lambda data: filtfilt(b, a, data)

def extract_envelope(signal, samplerate, filter_type='FIR', order=100):
    rectified_signal = np.abs(signal)  # Ensure rectification is correct
    lowpass_filter = create_lowpass_filter(400, samplerate, filter_type, order)
    return lowpass_filter(rectified_signal)


def plot_waveform_samplenumber(data, title):
    plt.figure(figsize=(10, 4))
    plt.plot(data)
    plt.title(title)
    plt.xlabel("Sample Number")
    plt.ylabel("Amplitude")
    plt.show()

data, samplerate = get_audio_info('downsampled_bohemian.wav')   

# Filter the sound with the passband bank
N = 10  # Adjust as needed
bands = define_frequency_bands(N)
filtered_signals = [create_bandpass_filter(band, samplerate)(data) for band in bands]

# Plot the output signals of the lowest and highest frequency channels
plot_waveform_samplenumber(filtered_signals[0], "Output Signal of the Lowest Frequency Channel")   
plot_waveform_samplenumber(filtered_signals[-1], "Output Signal of the Highest Frequency Channel")

# Rectify and extract envelopes
envelopes = [extract_envelope(signal, samplerate) for signal in filtered_signals]

# Plot the extracted envelopes
plot_waveform_samplenumber(envelopes[0], "Extracted Envelope of the Lowest Frequency Channel")   
plot_waveform_samplenumber(envelopes[-1], "Extracted Envelope of the Highest Frequency Channel")
