import pandas as pd
import data_cleaning as dc
import gradient_boosting as gb


def readData(i):
    X_train = pd.read_csv(f"data/X_Train{i}.csv", index_col=False)
    Y_train = pd.read_csv(f"data/Y_Train{i}.csv", index_col=False)
    X_test = pd.read_csv(f"data/X_Test{i}.csv", index_col=False)
    Y_test = pd.read_csv(f"data/Y_Test{i}.csv", index_col=False)
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    Y_train = Y_train["Credit_Limit"].to_numpy()
    Y_test = Y_test["Credit_Limit"].to_numpy()
    return X_train, X_test, Y_train, Y_test


if __name__ == '__main__':
    dataClean = dc.DataCleaning()
    dataClean.readData()

    result = ""

    r2 = 0
    mse = 0

    for i in range(5):
        X_train, X_test, Y_train, Y_test = readData(i)
        mseTemp, r2Temp = gb.gradientBoostingRegressionOutlier(X_train, X_test, Y_train, Y_test)
        result += f"round {i+1}\nMSE : {mseTemp}\nR2-Score : {r2Temp}\n----------------------------\n"
        r2 += r2Temp
        mse += mseTemp

    print(f"Average MSE : {mse/5}\nAverage R2 : {r2/5}")

    result += f"\nAverage MSE : {mse/5}\nAverage R2 : {r2/5}"

    f = open("result.txt", "a")
    f.write(result)
    f.close()
