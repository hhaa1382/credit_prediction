import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, SimpleImputer, _iterative
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
import miceforest as mcf
from missforest.miss_forest import MissForest


class DataCleaning:
    def __init__(self):
        self.data = None
        self.X = None
        self.Y = None

    def removeDuplicate(self):
        temp = pd.read_csv("CreditPrediction.csv")
        return temp.drop_duplicates()


    def checkCovMatrix(self, data, Y):
        print(np.cov(data, Y))


    def checkCorrelationScatter(self, feature, Y):
        plt.xlabel(feature)
        plt.ylabel("y")
        plt.scatter(self.data[feature], Y)
        plt.show()


    def checkOutlierMeanMode(self):
        self.data["Gender"].fillna(self.data["Gender"].mode()[0], inplace=True)
        self.data["Education_Level"].fillna(self.data["Education_Level"].mode()[0], inplace=True)
        self.data["Marital_Status"].fillna(self.data["Marital_Status"].mode()[0], inplace=True)
        self.data["Income_Category"].fillna(self.data["Income_Category"].mode()[0], inplace=True)
        self.data["Card_Category"].fillna(self.data["Card_Category"].mode()[0], inplace=True)
        self.data["Months_on_book"].fillna(self.data["Months_on_book"].mean(), inplace=True)
        self.data["Total_Relationship_Count"].fillna(self.data["Total_Relationship_Count"].mean(), inplace=True)
        self.data["Customer_Age"].fillna(self.data["Customer_Age"].mean(), inplace=True)
        self.Y.fillna(self.Y.mean(), inplace=True)


    def detect_outliers_iqr(self, data):
        outliers = []
        data = sorted(data)
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        IQR = q3 - q1
        lwr_bound = q1 - (1.5 * IQR)
        upr_bound = q3 + (1.5 * IQR)
        for i in data:
            if i < lwr_bound or i > upr_bound:
                outliers.append(i)
        return outliers



    def detect_outliers_zscore(self, data):
        outliers = []
        thres = 3

        mean = np.mean(data)
        std = np.std(data)

        for i in data:
            z_score = (i - mean) / std
            if (np.abs(z_score) > thres):
                outliers.append(i)
        return outliers



    def checkOutlierIQR(self):
        sample_outliers = self.detect_outliers_iqr(self.data["Avg_Utilization_Ratio"].to_numpy())

        for value in sample_outliers:
            self.data["Avg_Utilization_Ratio"].replace(value, np.nan, inplace=True)

        sample_outliers = self.detect_outliers_iqr(self.data["Total_Revolving_Bal"].to_numpy())

        for value in sample_outliers:
            self.data["Total_Revolving_Bal"].replace(value, np.nan, inplace=True)


    def checkOutlierZScore(self):
        sample_outliers = self.detect_outliers_zscore(self.data["Avg_Utilization_Ratio"].to_numpy())

        for value in sample_outliers:
            self.data["Avg_Utilization_Ratio"].replace(value, np.nan, inplace=True)

        sample_outliers = self.detect_outliers_zscore(self.data["Total_Revolving_Bal"].to_numpy())

        for value in sample_outliers:
            self.data["Total_Revolving_Bal"].replace(value, np.nan, inplace=True)


    def checkFeaturesCorrelation(self, Y_train):
        for col in self.data.columns:
            self.checkCovMatrix(self.data[col].to_numpy(), Y_train["Credit_Limit"].to_numpy())
            self.checkCorrelationScatter(col, Y_train)


    def checkAgeOutlier(self):
        holder = []
        for data in self.data["Customer_Age"]:
            if data > 150:
                holder.append(data)
        for age in holder:
            self.data["Customer_Age"].replace(age, np.nan, inplace=True)


    def imputeData(self):
        for i in range(5):
            X_train, X_test, Y_train, Y_test = train_test_split(self.data, self.Y, test_size=0.3)
            Y_train = pd.DataFrame(Y_train, columns=["Credit_Limit"])
            Y_test = pd.DataFrame(Y_test, columns=["Credit_Limit"])

            columns = ["Total_Revolving_Bal", "Avg_Utilization_Ratio", "Income_Category",
                       "Card_Category"]

            # scaler = StandardScaler()
            # X_train = scaler.fit_transform(X_train[columns])
            # X_test = scaler.transform(X_test[columns])

            X_train = pd.DataFrame(X_train, columns=columns)
            X_test = pd.DataFrame(X_test, columns=columns)

            # X_train_copy = X_train.copy()
            # X_test_copy = X_test.copy()

            # self.data["Credit_Limit"] = Y_train.Credit_Limit
            # X_test["Credit_Limit"] = 0

            # self.checkAgeOutlier()
            # self.checkOutlierIQR()
            # self.checkOutlierZScore()
            # self.checkOutlierMeanMode()

            # X_train_copy["Avg_Utilization_Ratio"] = X_train_copy["Avg_Utilization_Ratio"] * 1000
            # X_test_copy["Avg_Utilization_Ratio"] = X_test_copy["Avg_Utilization_Ratio"] * 1000
            # X_train_copy["Total_Revolving_Bal"] = X_train_copy["Total_Revolving_Bal"] * 5
            # X_test_copy["Total_Revolving_Bal"] = X_test_copy["Total_Revolving_Bal"] * 5

            # numericalImputer = SimpleImputer(missing_values=np.nan, strategy="mean")
            # numericDFTrain = pd.DataFrame(numericalImputer.fit_transform(self.data), columns=self.data.columns)
            # numericDFTest = pd.DataFrame(numericalImputer.transform(X_test), columns=self.data.columns)


            regressor = LinearRegression()
            # regressor = RandomForestRegressor(n_estimators=100, max_depth=11)

            numericalImputer = _iterative.IterativeImputer(estimator=regressor, max_iter=400,
                                                           imputation_order='roman', tol=1e-5,
                                                           missing_values=np.nan, n_nearest_features=1)
            numericDFTrain = pd.DataFrame(numericalImputer.fit_transform(X_train), columns=columns)
            numericDFTest = pd.DataFrame(numericalImputer.transform(X_test), columns=columns)


            # numericalImputer = KNNImputer(n_neighbors=24)
            # numericDFTrainKNN = pd.DataFrame(numericalImputer.fit_transform(X_train_copy), columns=columns)
            # numericDFTestKNN = pd.DataFrame(numericalImputer.transform(X_test_copy), columns=columns)
            #
            # numericDFTrainKNN["Avg_Utilization_Ratio"] = numericDFTrainKNN["Avg_Utilization_Ratio"] / 1000
            # numericDFTestKNN["Avg_Utilization_Ratio"] = numericDFTestKNN["Avg_Utilization_Ratio"] / 1000
            # numericDFTrainKNN["Total_Revolving_Bal"] = numericDFTrainKNN["Total_Revolving_Bal"] / 5
            # numericDFTestKNN["Total_Revolving_Bal"] = numericDFTestKNN["Total_Revolving_Bal"] / 5

            # self.checkFeaturesCorrelation(Y_train)

            newDFTrain = pd.DataFrame()
            newDFTest = pd.DataFrame()

            # newDFTrain["Total_Relationship_Count"] = numericDFTrain.Total_Relationship_Count
            # newDFTest["Total_Relationship_Count"] = numericDFTest.Total_Relationship_Count

            # newDFTrain["Total_Amt_Chng_Q4_Q1"] = numericDFTrain.Total_Amt_Chng_Q4_Q1
            # newDFTest["Total_Amt_Chng_Q4_Q1"] = numericDFTest.Total_Amt_Chng_Q4_Q1

            # newDFTrain["Total_Trans_Amt"] = numericDFTrain.Total_Trans_Amt
            # newDFTest["Total_Trans_Amt"] = numericDFTest.Total_Trans_Amt

            newDFTrain["Total_Revolving_Bal"] = numericDFTrain.Total_Revolving_Bal
            newDFTest["Total_Revolving_Bal"] = numericDFTest.Total_Revolving_Bal

            newDFTrain["Avg_Utilization_Ratio"] = numericDFTrain.Avg_Utilization_Ratio
            newDFTest["Avg_Utilization_Ratio"] = numericDFTest.Avg_Utilization_Ratio

            newDFTrain["Income_Category"] = numericDFTrain.Income_Category
            newDFTest["Income_Category"] = numericDFTest.Income_Category

            newDFTrain["Card_Category"] = numericDFTrain.Card_Category
            newDFTest["Card_Category"] = numericDFTest.Card_Category

            # newDFTrainKNN = pd.DataFrame()
            # newDFTestKNN = pd.DataFrame()

            # newDFTrainKNN["Total_Revolving_Bal"] = numericDFTrainKNN.Total_Revolving_Bal
            # newDFTestKNN["Total_Revolving_Bal"] = numericDFTestKNN.Total_Revolving_Bal
            #
            # newDFTrainKNN["Avg_Utilization_Ratio"] = numericDFTrainKNN.Avg_Utilization_Ratio
            # newDFTestKNN["Avg_Utilization_Ratio"] = numericDFTestKNN.Avg_Utilization_Ratio
            #
            # newDFTrainKNN["Income_Category"] = numericDFTrainKNN.Income_Category
            # newDFTestKNN["Income_Category"] = numericDFTestKNN.Income_Category
            #
            # newDFTrainKNN["Card_Category"] = numericDFTrainKNN.Card_Category
            # newDFTestKNN["Card_Category"] = numericDFTestKNN.Card_Category

            newDFTrain.to_csv(f"data/X_Train{i}.csv", index=False)
            newDFTest.to_csv(f"data/X_Test{i}.csv", index=False)

            # newDFTrainKNN.to_csv(f"data/X_Train_KNN{i}.csv", index=False)
            # newDFTestKNN.to_csv(f"data/X_Test_KNN{i}.csv", index=False)

            Y_train.to_csv(f"data/Y_Train{i}.csv", index=False)
            Y_test.to_csv(f"data/Y_Test{i}.csv", index=False)


    def encodingData(self):
        self.data["Gender"].replace(["F", "M"], [1, 2], inplace=True)

        self.data["Education_Level"].replace("Unknown", np.nan, inplace=True)
        self.data["Education_Level"].replace(["Uneducated", "High School", "Graduate",
                                              "College", "Post-Graduate", "Doctorate"], [1, 2, 3, 4, 5, 6],
                                             inplace=True)

        self.data["Marital_Status"].replace("Unknown", np.nan, inplace=True)
        self.data["Marital_Status"].replace(["Single", "Married", "Divorced"], [1, 2, 3], inplace=True)

        self.data["Income_Category"].replace("Unknown", np.nan, inplace=True)
        self.data["Income_Category"].replace(["Less than $40K", "$40K - $60K", "$60K - $80K",
                                              "$80K - $120K", "$120K +"], [1, 2, 3, 4, 5], inplace=True)

        self.data["Card_Category"].replace(["Blue", "Gold", "Silver", "Platinum"],
                                           [1, 2, 3, 4], inplace=True)

        self.imputeData()


    def readData(self):
        self.data = self.removeDuplicate()
        self.Y = pd.DataFrame(self.data["Credit_Limit"])
        self.data = self.data.drop(["CLIENTNUM", "Unnamed: 19", "Credit_Limit"], axis=1)
        self.encodingData()

