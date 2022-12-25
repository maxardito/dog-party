import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import librosa
import numpy as np
import pyaudio
import time
import feature_extractor
import evaluation
import preprocess

# Hyperparameters
CHANNELS = 1
BLOCK_SIZE = 1024
RATE = 44100
WINDOW_SIZE = 30
HOP_SIZE = 5
NUM_MFCCS = 13

NUM_SPEAKERS = 3
DRIVES = ['./files/sam/', './files/sal', './files/hrh/']

# Audio callback
p = pyaudio.PyAudio()

# Global variable for features to be sent from
# audio callback to KNN function
final_data = np.ones((1, 14))

# Define the feature extractor used for training and implementation
extractor = feature_extractor.FeatureExtractor(WINDOW_SIZE, HOP_SIZE,
                                               NUM_MFCCS, RATE)
"""
Audio callback function accesses the final_data global variable
and extracts a new block of feature vectors for each audio I/O vector
"""


def callback(in_data, frame_count, time_info, flag):
    # Access global variable
    global final_data

    # Convert input data to float vector and normalize
    data = np.frombuffer(in_data, dtype=np.float32)

    # Extract the features
    (pitch, mfccs, size) = extractor.extract_features(data)

    # Throw away data below a certain amplitude threshold
    volume_norm = np.linalg.norm(data)

    if (size == 0 or volume_norm < 0.4):
        final_data = "Unvoiced"
    else:
        # Assign first feature to f0 and remaining features to mfccs
        new_data = np.ones((size, 14))

        new_data[:, 0] = pitch

        for i in range(13):
            new_data[:, (i + 1)] = mfccs[i, :]

        final_data = new_data

    return (data, pyaudio.paContinue)


def main(**kwargs):
    # Define training drives
    preprocessor = preprocess.Preprocessor(extractor, DRIVES, NUM_SPEAKERS,
                                           RATE)

    # Load CSV file and extract features and classes
    df = pd.read_csv('./data.csv')
    # df = preprocessor.process_audio_files()
    data = df.to_numpy()

    # Define feature matrix and class vector
    X = data[:, :(NUM_MFCCS + 1)]
    y = data[:, (NUM_MFCCS + 1)].astype(int)

    # Evaluate the performance of the model
    evaluation.evaluate(X, y)

    # Normalize data
    normalizer = preprocessing.Normalizer().fit(X)
    X_norm = normalizer.transform(X)

    # Train the classifier on the entire dataset
    neighbor = KNeighborsClassifier(n_neighbors=5)
    neighbor.fit(X_norm, y)

    # Start the audio callback
    stream = p.open(format=pyaudio.paFloat32,
                    channels=CHANNELS,
                    rate=RATE,
                    output=False,
                    input=True,
                    frames_per_buffer=BLOCK_SIZE,
                    stream_callback=callback)

    stream.start_stream()

    ###### Audio Callback ######
    while stream.is_active():
        time.sleep(0.01)
        # Filter out NaN data
        if (isinstance(final_data, str)):
            # print(final_data)
            continue
        else:
            # Normalize and predict new vector of windows
            X_new = normalizer.fit_transform(final_data)
            result = neighbor.predict(X_new)

            # Print the most frequently recurring speaker estimate
            values, counts = np.unique(result, return_counts=True)
            print(values[counts.argmax()])

            # Stop the stream
            # stream.stop_stream()
    stream.close()

    p.terminate()


if __name__ == "__main__":
    main()
