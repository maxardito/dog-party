import librosa
import numpy as np
"""
FeatureExtractor

Extracts necessary features for the speaker ID model. Constructor 
takes the following arguments...

window_size: Analysis window size in milliseconds
hop_size: Analysis window hop size in milliseconds
num_mfccs: Number of MFCCs to extract
sr: Sampling rate
"""


class FeatureExtractor:

    def __init__(self, window_size, hop_size, num_mfccs, sr):
        self.window_size = window_size
        self.hop_size = hop_size
        self.num_mfccs = num_mfccs
        self.sr = sr

    def extract_features(self, signal):
        windowLength = int((self.window_size / 1000) * self.sr)
        hopLength = int((self.hop_size / 1000) * self.sr)

        f0, voiced_flag, voiced_probs = librosa.pyin(
            y=signal,
            win_length=windowLength,
            hop_length=hopLength,
            fmin=librosa.note_to_hz('C1'),
            fmax=librosa.note_to_hz('C4'))
        total_analysis_windows = f0.shape[0]

        voiced_flag = np.where(voiced_flag == True, 1, 0)

        f0 = f0 * voiced_flag

        f0_voiced = np.empty(0)

        for i in range(total_analysis_windows):
            if (voiced_flag[i] == 1):
                f0_voiced = np.append(f0_voiced, f0[i])

        mfccs = librosa.feature.mfcc(y=signal,
                                     n_mfcc=self.num_mfccs,
                                     win_length=windowLength,
                                     hop_length=hopLength,
                                     sr=self.sr,
                                     n_fft=windowLength)

        mfccs = mfccs * voiced_flag

        mfccs_voiced = np.zeros((13, np.sum(voiced_flag)))

        window_counter = 0

        for i in range(total_analysis_windows):
            if (voiced_flag[i] == 1):
                mfccs_voiced[:, window_counter] = mfccs[:, i]
                window_counter += 1

        return (f0_voiced, mfccs_voiced, window_counter)
