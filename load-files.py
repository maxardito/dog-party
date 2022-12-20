import librosa
import librosa.display
import numpy as np
import pandas as pd
from extract_data import extract_training_data

thatcher_drive = "16000_pcm_speeches/Magaret_Tarcher/"
mandela_drive = "16000_pcm_speeches/Nelson_Mandela/"
netanyahu_drive = "16000_pcm_speeches/Benjamin_Netanyau/"

drives = [thatcher_drive, mandela_drive, netanyahu_drive]

num_files = 1499
num_speakers = 3

windowLength = 30
hopLength = 5

num_coeff = 13
data = np.empty((14, 0))

# dictionary of lists
dict = {
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

df = pd.DataFrame(dict)

for i in range(num_speakers):
    for j in range(num_files):
        print("File: ", j, ", Speaker: ", i)
        signal, sr = librosa.load(drives[i] + str(j) + ".wav")
        (pitch, mfccs) = extract_training_data(signal, sr, windowLength,
                                               hopLength, num_coeff)

        speaker = np.ones(pitch.shape[0]) * i

        new_data = np.row_stack((pitch, mfccs, speaker))
        new_data = pd.DataFrame(new_data.transpose(), columns=df.columns)

        df = df.append(new_data, ignore_index=True)

csv_data = df.to_csv('speakers.csv', index=False)
