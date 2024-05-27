from xgboost import XGBRegressor, XGBRFRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score


def otherRegressors(X_train, X_test, Y_train, Y_test):
    regressors = {
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(max_depth=9),
        "XGBoost": XGBRFRegressor()
    }

    for name, model in regressors.items():
        model.fit(X_train, Y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(Y_test, predictions)
        r2 = r2_score(Y_test, predictions)
        print(f"{name} Mean Squared Error: {mse}")
        print(f"{name} R2-Score: {r2}")
        print("-------------------------------")
