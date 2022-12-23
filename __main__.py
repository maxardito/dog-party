import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import librosa
import numpy as np
import pyaudio
import time

CHANNELS = 2
CHUNK = 1024
RATE = 44100 

final_data = np.ones((1, 14))
p = pyaudio.PyAudio()

min_max_training = np.zeros((2, 14))

def callback(in_data, frame_count, time_info, flag):
    global final_data

    data = np.frombuffer(in_data, dtype=np.float32)
    volume_norm = np.linalg.norm(data)

    # Extract using librosa
    (pitch, mfccs, size) = extract_training_data(data, 44100, 30, 5, 13)

    if(size == 0 or volume_norm < 0.4):
      final_data = "Unvoiced" 
    else: 
      new_data = np.ones((size, 14))

      new_data[:, 0] = pitch

      for i in range(13):
        new_data[:, (i + 1)] = mfccs[i, :]

      # Clipping
      # for i in range(14):
      #   if(np.min(new_data[:, i]) < min_max_training[0, i] or np.max(new_data[:, i]) < min_max_training[1, i]):
      #     final_data = "Unvoiced"
 
      # Extra zerox
      zcr = librosa.feature.zero_crossing_rate(y=data, frame_length=2048,hop_length=5)
      if(np.average(zcr) < 0.02):
        final_data = "Unvoiced"

      final_data = new_data

    return (data, pyaudio.paContinue)

def extract_training_data(signal, sr, windowLength, hopLength, num_coeff):
  windowLength = int((windowLength / 1000) * sr)
  hopLength = int((hopLength / 1000) * sr)
 
  f0, voiced_flag, voiced_probs = librosa.pyin(y=signal, win_length=windowLength, hop_length=hopLength, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C4'))
  total_analysis_windows = f0.shape[0]
  
  voiced_flag = np.where(voiced_flag == True, 1, 0)
  
  f0 = f0 * voiced_flag
  
  f0_voiced = np.empty(0)
  
  for i in range(total_analysis_windows):
    if(voiced_flag[i] == 1):
      f0_voiced = np.append(f0_voiced, f0[i]) 

  mfccs = librosa.feature.mfcc(y=signal, n_mfcc=num_coeff, win_length=windowLength, hop_length=hopLength, sr=sr, n_fft=windowLength)
  
  mfccs = mfccs * voiced_flag
  
  # NOTE: do I neeed the second one?
  mfccs_voiced = np.empty((13, np.sum(voiced_flag)))
  mfccs_voiced = np.zeros((13, np.sum(voiced_flag)))

  window_counter = 0

  for i in range(total_analysis_windows):
    if(voiced_flag[i] == 1):
      mfccs_voiced[:, window_counter] = mfccs[:, i] 
      window_counter += 1

  
  return (f0_voiced, mfccs_voiced, window_counter)


def evaluate(X, y):
    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        random_state=7,
                                                        test_size=0.01,
                                                        shuffle=True)

    normalizer = preprocessing.Normalizer().fit(X_train)

    X_norm_train = normalizer.transform(X_train)

    neighbor = KNeighborsClassifier(n_neighbors=5)

    neighbor.fit(X_norm_train, y_train)

    X_norm_test = normalizer.transform(X_test)
    pred = neighbor.predict(X_norm_test)
    print(pred)

    accuracy = neighbor.score(X_test, y_test, sample_weight=None)
    print("Accuracy: ", accuracy * 100, "%")


def main(**kwargs):
    num_features = 14

    # Load CSV file and extract features and classes
    df = pd.read_csv('./speakers.csv')
    data = df.to_numpy()
    X = data[:, :num_features]
    y = data[:, num_features].astype(int)

    # evaluate(X, y)

    # Normalize data
    normalizer = preprocessing.Normalizer().fit(X)

    X_norm = normalizer.transform(X)

    neighbor = KNeighborsClassifier(n_neighbors=5)

    neighbor.fit(X_norm, y)

    for i in range(14):
      min_max_training[0, i] = np.min(X[i, :])
      min_max_training[1, i] = np.max(X[i, :])

    stream = p.open(format=pyaudio.paFloat32,
                channels=CHANNELS,
                rate=RATE,
                output=False,
                input=True,
                frames_per_buffer = CHUNK,
                stream_callback=callback)

    stream.start_stream()

    while stream.is_active():
        time.sleep(0.01)
        if(isinstance(final_data, str)):
          # print(final_data)
          continue
        else:
          X_new = normalizer.fit_transform(final_data)
          result = neighbor.predict(X_new)
          values, counts = np.unique(result, return_counts=True)

          print(result)
          # print(values[counts.argmax()])
          # stream.stop_stream()
    stream.close()

    p.terminate()


if __name__ == "__main__":
    main()
