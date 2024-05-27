import keras.src.activations
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def neuralNetworkRegression(X_train, X_test, Y_train, Y_test):
    # norm = layers.Normalization(input_shape=[1, ], axis=None)
    # data = norm.adapt(data)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    regression = Sequential()
    regression.add(Dense(40, activation="relu", input_shape=(4,)))
    regression.add(Dense(50, activation="relu"))
    # regression.add(Dense(50, activation="relu"))
    regression.add(Dense(1, activation="linear"))
    regression.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adam(learning_rate=0.1))

    # early_stopping = [EarlyStopping(monitor="val_loss", patience=10, min_delta=1e-6)]

    regression.fit(X_train, Y_train, epochs=60, validation_split=0.2)
    Y_predict = regression.predict(X_test)
    mse = mean_squared_error(Y_test, Y_predict)
    r2 = r2_score(Y_test, Y_predict)
    print(f"MSE : {mse}")
    print(f"r2-score : {r2}")
