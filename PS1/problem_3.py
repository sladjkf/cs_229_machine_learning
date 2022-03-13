import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def main():
    df_train = pd.read_csv("data/ds4_train.csv")
    df_valid = pd.read_csv("data/ds4_valid.csv")


    X_train = df_train.loc[:,df_train.columns != "y"]

    PR = PoissonRegression()
    PR.train(df_train.loc[:,df_train.columns != "y"],df_train["y"],alpha=0.1)

    preds = PR.predict(X_train)

    preds=np.array([np.exp(X_mat[i] @ PR.theta) for i in range(0,2500)])

    plt.scatter(df_train["y"],preds,marker='x')
    plt.plot(np.linspace(df_train["y"].min(),df_train["y"].max()),np.linspace(df_train["y"].min(),df_train["y"].max()))
    plt.xlabel("actual")
    plt.ylabel("predicted")

class PoissonRegression:
    def __init__(self):
        self.theta = 0
    def train(self,X,Y,eps = 1e-5,alpha=1e-8,iters=15):
        m = X.shape[0]
        n = X.shape[1]

        # add the constant weight column
        X_mat = np.c_[np.ones(m),np.array(X)]
        X_mat_t = X_mat.T
        Y_vec = np.array(Y)

        self.theta = np.zeros(n+1)

        err = np.inf*np.ones(n+1)

        #while np.linalg.norm(err,ord=1) >= eps:
        #    preds = self.predict(X_mat,cast=False)
        #    gradient = X_mat_t @ (Y_vec - preds)
        #    print(gradient)
        #    gradient = np.array([sum(X_mat[i][j]*(Y_vec[i]-preds[i]) for i in range(0,m)) for j in range(0,n+1)])
        #    print(gradient)
        #    thet_new = self.theta + alpha*gradient
        #    err = thet_new - self.theta
        #    self.theta=thet_new
        #    print(np.linalg.norm(gradient,ord=1))

        ## SGD
        for iter in range(0,iters):
            #grad_avg = np.zeros(n+1)
            for i in range(0,m):
                h_i = np.exp(self.theta.T @ X_mat[i])
                gradient = X_mat[i]*(Y_vec[i]-h_i)
                self.theta = self.theta + alpha*gradient
                print(self.theta)
            #print(grad_avg/m)


    def predict(self,X,cast=True):
        if cast:
            X_mat = np.c_[np.ones(X.shape[0]),np.array(X)]
        else:
            X_mat = X

        return np.exp(X_mat @ self.theta)
