from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def gradientBoostingRegressionOutlier(X_train, X_test, Y_train, Y_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    regressor = GradientBoostingRegressor(
        max_depth=7
    )

    kf = KFold(n_splits=5, shuffle=True)
    cv_score = cross_val_score(regressor, X_train, Y_train, cv=kf)
    print(f"cross val scores : {cv_score}")

    regressor.fit(X_train, Y_train)
    Y_predict = regressor.predict(X_test)
    mse = mean_squared_error(Y_test, Y_predict)
    r2 = r2_score(Y_test, Y_predict)
    print(f"MSE : {mse}")
    print(f"r2-score : {r2}")
    print("--------------------------------")

    return mse, r2

