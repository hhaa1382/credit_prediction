from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def randomForestRegression(X_train, X_test, Y_train, Y_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    for i in range(100, 200, 10):
        regressor = RandomForestRegressor(n_estimators=i, max_depth=11, min_samples_leaf=2)
        regressor.fit(X_train, Y_train)
        Y_predict = regressor.predict(X_test)
        mse = mean_squared_error(Y_test, Y_predict)
        r2 = r2_score(Y_test, Y_predict)
        print(f"i = {i}")
        print(f"MSE : {mse}")
        print(f"r2-score : {r2}")
