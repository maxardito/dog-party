import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import sounddevice as sd

duration = 10  #in seconds


def audio_callback(indata, frames, time, status):
    volume_norm = np.linalg.norm(indata) * 10
    # print("|" * int(volume_norm))
    print(indata.shape)


def evaluate(X, y):
    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        random_state=42,
                                                        test_size=0.01,
                                                        shuffle=True)

    neighbor = KNeighborsClassifier(n_neighbors=5)

    neighbor.fit(X_train, y_train)
    neighbor.predict(X_test)

    accuracy = neighbor.score(X_test, y_test, sample_weight=None)
    print("Accuracy: ", accuracy * 100, "%")


def run(X, y):
    neighbor = KNeighborsClassifier(n_neighbors=5)

    neighbor.fit(X, y)

    prediction = neighbor.predict([X[4]])
    stream = sd.InputStream(callback=audio_callback, blocksize=256)
    with stream:
        sd.sleep(duration * 1000)


def main():
    num_features = 14

    # Load CSV file and extract features and classes
    df = pd.read_csv('./speakers.csv')
    data = df.to_numpy()
    X = data[:, :num_features]
    y = data[:, num_features].astype(int)

    # Normalize data
    X_norm = np.ndarray(X.shape)

    for i in range(X.shape[1]):
        X_norm[:, i] = X[:, i] / np.linalg.norm(X[:, i])

    # evaluate(X_norm, y)

    run(X_norm, y)


if __name__ == "__main__":
    main()
