import librosa
import numpy as np


class FeatureExtractor:

    def __init__(self, window_size, hop_size, num_mfccs):
        self.window_size = window_size
        self.hop_size = hop_size
        self.num_mfccs = num_mfccs

    def extract_f0(self, signal, sr):
        # Calculate window size and hop size based on current audio file's sample rate
        current_window_size = int((self.window_size / 1000) * sr)
        current_hop_size = int((self.hop_size / 1000) * sr)

        # Calculate f0
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y=signal,
            win_length=current_window_size,
            hop_length=current_hop_size,
            fmin=librosa.note_to_hz('C1'),
            fmax=librosa.note_to_hz('C7'))

        self.total_analysis_windows = f0.shape[0]

        self.voiced_flag = np.where(voiced_flag == True, 1, 0)

        f0 = f0 * voiced_flag

        f0_voiced = np.empty(0)

        for i in range(self.total_analysis_windows):
            if (voiced_flag[i] == 1):
                f0_voiced = np.append(f0_voiced, f0[i])

        return f0_voiced

    def extract_mfccs(self, signal, sr):
        # Calculate window size and hop size based on current audio file's sample rate
        current_window_size = int((self.window_size / 1000) * sr)
        current_hop_size = int((self.hop_size / 1000) * sr)

        # FIX: f0 function must be called before mfcc function
        mfccs = librosa.feature.mfcc(y=signal,
                                     n_mfcc=self.num_mfccs,
                                     win_length=current_window_size,
                                     hop_length=current_hop_size,
                                     sr=sr)

        mfccs = mfccs * self.voiced_flag

        mfccs_voiced = np.zeros((13, np.sum(self.voiced_flag)))

        window_counter = 0
        for i in range(self.total_analysis_windows):
            if (self.voiced_flag[i] == 1):
                mfccs_voiced[:, window_counter] = mfccs[:, i]
                window_counter += 1

        return mfccs_voiced
