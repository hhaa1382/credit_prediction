from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def linearRegression(X_train, X_test, Y_train, Y_test):
    poly = PolynomialFeatures(degree=2)
    X_train = poly.fit_transform(X_train)
    X_test = poly.fit_transform(X_test)
    regression = LinearRegression()
    regression.fit(X_train, Y_train)
    Y_predict = regression.predict(X_test)
    mse = mean_squared_error(Y_test, Y_predict)
    r2 = r2_score(Y_test, Y_predict)
    print(f"MSE : {mse}")
    print(f"r2-score : {r2}")


