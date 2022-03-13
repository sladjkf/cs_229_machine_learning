import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def main():
    df_train = pd.read_csv("data/ds5_train.csv")
    df_valid = pd.read_csv("data/ds5_valid.csv")
    df_test = pd.read_csv("data/ds5_test.csv")

    LWR = LocWeightRegression()
    LWR.fit(df_train[["x_1"]],df_train["y"])

    valid_preds = LWR.predict(df_valid[["x_1"]])
    N = len(df_valid["y"])
    print("validation mse: ", (1/N)*sum((df_valid["y"]-valid_preds)**2))

    plt.scatter(x="x_1",y="y",data=df_train,marker="x")
    plt.scatter(df_train["x_1"],LWR.predict(df_train[["x_1"]]))

    plt.scatter(x="x_1",y="y",data=df_valid,marker="x")
    plt.scatter(df_valid["x_1"],LWR.predict(df_valid[["x_1"]]))

    plt.scatter(x="x_1",y="y",data=df_test,marker="x")
    plt.scatter(df_test["x_1"],LWR.predict(df_test[["x_1"]]))

    tau_values = [0.03,0.05,0.1,0.5,1,10]
    for tau in tau_values:
        LWR = LocWeightRegression()
        LWR.fit(df_train[["x_1"]],df_train["y"])
        valid_preds = LWR.predict(df_valid[["x_1"]],tau=tau)
        print("validation set mse for tau={}: {}".format(tau,(1/N)*sum((df_valid["y"]-valid_preds)**2)))

        plt.figure()
        plt.scatter(x="x_1",y="y",data=df_valid,marker="x")
        plt.scatter(df_valid["x_1"],valid_preds)
        plt.title("tau="+str(tau))
        plt.show()





class LocWeightRegression:
    def __init__(self):
        self.theta = None
    def fit(self,X,Y):
        m = X.shape[0]
        n = X.shape[1]

        # add the constant weight column
        X_mat = np.c_[np.ones(m),np.array(X)]
        X_mat_t = X_mat.T

        self.m = m
        self.n = n
        self.X = X_mat
        self.X_t = X_mat_t
        self.Y = np.array(Y)

    def predict(self,X,cast=True,tau=0.5):
        def single_pred(x):
            W = np.diag([np.exp(-(1/tau**2)*np.linalg.norm(self.X[i]-x,ord=2)**2) for i in range(0,self.m)])
            theta = np.linalg.inv(self.X_t @ W @ self.X) @ self.X_t @ W @ self.Y
            return theta @ x
        if (len(X.shape) == 1):
            return single_pred(X)
        elif (len(X.shape) == 2):
            if cast:
                X_mat = np.c_[np.ones(X.shape[0]),np.array(X)]
            else:
                X_mat = X
            return np.array([single_pred(X_mat[i]) for i in range(0,X_mat.shape[0])])


