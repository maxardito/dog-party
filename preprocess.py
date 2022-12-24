import librosa
import librosa.display
import numpy as np
import pandas as pd


class Preprocessor:

    def __init__(self, extractor, drives, num_files, num_speakers):
        self.extractor = extractor
        self.drives = drives
        self.num_files = num_files
        self.num_speakers = num_speakers

        self.data = np.empty((14, 0))

        # Dictionary of lists
        feature_table = {
            'f0': [],
            'MFCC: 1': [],
            'MFCC: 2': [],
            'MFCC: 3': [],
            'MFCC: 4': [],
            'MFCC: 5': [],
            'MFCC: 6': [],
            'MFCC: 7': [],
            'MFCC: 8': [],
            'MFCC: 9': [],
            'MFCC: 10': [],
            'MFCC: 11': [],
            'MFCC: 12': [],
            'MFCC: 13': [],
            'Speaker': []
        }

        self.df = pd.DataFrame(feature_table)

    def process_audio_files(self):

        for i in range(self.num_speakers):
            for j in range(self.num_files):
                print("File: ", j, ", Speaker: ", i)
                signal, sr = librosa.load(self.drives[i] + str(j) + ".wav")
                (pitch, mfccs) = self.extractor.extract_training_data(signal)

                speaker = np.ones(pitch.shape[0]) * i

                new_data = np.row_stack((pitch, mfccs, speaker))
                new_data = pd.DataFrame(new_data.transpose(),
                                        columns=self.df.columns)

                self.df = self.df.append(new_data, ignore_index=True)

        return self.df
