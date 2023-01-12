from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


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
