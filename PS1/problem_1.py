import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

if __name__=="__main__":
    main()

def main():
    df_1_train = pd.read_csv("data/ds1_train.csv")
    df_2_train = pd.read_csv("data/ds2_train.csv")

    df_1_valid = pd.read_csv("data/ds1_valid.csv")
    df_2_valid = pd.read_csv("data/ds2_valid.csv")

    LR_1 = LogisticRegression()
    LR_1.train(df_1_train[["x_1","x_2"]],df_1_train["y"])
    LR_1_preds = LR_1.predict(df_1_train[["x_1","x_2"]],prob=False)
    LR_1_preds_v = LR_1.predict(df_1_valid[["x_1","x_2"]],prob=False)

    GDA_1 = GDA()
    GDA_1.train(df_1_train[["x_1","x_2"]],df_1_train["y"])
    GDA_1_preds = GDA_1.predict(df_1_train[["x_1","x_2"]],prob=False)
    GDA_1_preds_v = GDA_1.predict(df_1_valid[["x_1","x_2"]],prob=False)

    plot_model(df_1_train,np.append(np.array(GDA_1.thet_0),GDA_1.thet),"Set 1: GDA")
    plot_model(df_1_train,LR_1.theta,"Set 2: LR")

    print("set 1: train")
    print("LR\n",confusion_mat(df_1_train["y"],LR_1_preds))
    print("GDA\n",confusion_mat(df_1_train["y"],GDA_1_preds))

    print("set 1: validation")
    print("LR\n",confusion_mat(df_1_valid["y"],LR_1_preds_v))
    print("GDA\n",confusion_mat(df_1_valid["y"],GDA_1_preds_v))

    # -------

    LR_2 = LogisticRegression()
    LR_2.train(df_2_train[["x_1","x_2"]],df_2_train["y"])
    LR_2_preds = LR_2.predict(df_2_train[["x_1","x_2"]],prob=False)
    LR_2_preds_v = LR_2.predict(df_2_valid[["x_1","x_2"]],prob=False)

    GDA_2 = GDA()
    GDA_2.train(df_2_train[["x_1","x_2"]],df_2_train["y"])
    GDA_2_preds = GDA_2.predict(df_2_train[["x_1","x_2"]],prob=False)
    GDA_2_preds_v = GDA_2.predict(df_2_valid[["x_1","x_2"]],prob=False)


    plot_model(df_2_train,np.append(np.array(GDA_2.thet_0),GDA_2.thet),"Set 2: GDA")
    plot_model(df_2_train,LR_2.theta,"Set 2: LR")

    print("set 2: train")
    print("LR\n",confusion_mat(df_2_train["y"],LR_2_preds))
    print("GDA\n",confusion_mat(df_2_train["y"],GDA_2_preds))

    print("set 2: validation")
    print("LR\n",confusion_mat(df_2_valid["y"],LR_2_preds_v))
    print("GDA\n",confusion_mat(df_2_valid["y"],GDA_2_preds_v))


def confusion_mat(y_true,y_preds):
    true_pos = sum(x[0] & x[1] for x in zip(y_true==1,y_preds==1))
    true_neg = sum(x[0] & x[1] for x in zip(y_true==0,y_preds==0))
    false_pos = sum(x[0] & x[1] for x in zip(y_true==0,y_preds==1))
    false_neg = sum(x[0] & x[1] for x in zip(y_true==1,y_preds==0))

    return np.array([[true_pos,false_neg],[false_pos,true_neg]])

def plot_model(train,theta,title=""):
    fig,ax = plt.subplots()
    pos = train[train["y"]==1]
    neg = train[train["y"]==0]
    ax.scatter(pos["x_1"],pos["x_2"],marker='o')
    ax.scatter(neg["x_1"],neg["x_2"],marker='x')

    boundary_x = np.linspace(min(train["x_1"]),max(train["x_1"]))
    ax.plot(boundary_x,(-theta[0] - theta[1]*boundary_x)/theta[2])
    fig.suptitle(title)
    return fig

class LogisticRegression:
    def __init__(self):
        self.theta = np.zeros(1)

    def train(self,X,Y,eps=1e-5):
        """
        X: m by n, features. as pandas dataframe 
        Y: m by 1, label vector. as pandas series
        return: n+1 dimension, parameter vector theta
        """

        m = X.shape[0]
        n = X.shape[1]

        # add the constant weight column
        X_mat = np.c_[np.ones(m),np.array(X)]
        X_mat_t = X_mat.T
        #X_mat = np.array(X_modified)
        Y_vec = np.array(Y)

        self.theta = np.zeros(n+1)

        err = np.inf*np.ones(n+1)

        while np.linalg.norm(err,ord=1) >= eps:
            h_x = self.predict(X_mat,cast=False)
            hess = X_mat_t @ np.diag(h_x*(1-h_x)) @ X_mat
            hess_inv = np.linalg.inv(hess)
            grad = (-1/m)*sum((Y_vec[i] - h_x[i])*X_mat[i] for i in range(0,m))
            thet_new = self.theta - hess_inv @ grad
            err = thet_new - self.theta 
            self.theta = thet_new

    def predict(self,X,cast=True,prob=True):
        if cast:
            X_mat = np.c_[np.ones(X.shape[0]),np.array(X)]
        else:
            X_mat = X
        #print(X_modified)
        probs = 1 / (1 + np.exp(-self.theta @ X_mat.T))
        if prob:
            return probs
        else:
            return np.int64(probs >= 0.5)

class GDA:
    def __init__(self):
        self.phi = 0
        self.mu_0 = 0 
        self.mu_1 = 0
        self.sigma = 0
        self.sigma_inv = 0
        self.thet = 0
        self.thet_0 = 0

    def train(self,X,Y):
        m = X.shape[0]
        n = X.shape[1]

        X_mat=np.array(X)
        Y_vec = np.array(Y)

        self.phi = (1/m)*sum(Y)
        self.mu_0 = sum(int(Y_vec[i]==0)*X_mat[i] for i in range(0,m))/sum(Y_vec[i]==0 for i in range(0,m))

        self.mu_1 = sum(int(Y_vec[i]==1)*X_mat[i] for i in range(0,m))/sum(Y_vec[i]==1 for i in range(0,m))

        sigma = np.zeros((n,n))
        for i in range(0,m):
            mu_yi = {0:self.mu_0, 1:self.mu_1}[int(Y_vec[i])]
            sigma += np.outer((X_mat[i]-mu_yi),(X_mat[i]-mu_yi))
        self.sigma = (1/m)*sigma
        self.sigma_inv = np.linalg.inv(self.sigma)

        self.thet = (self.mu_1-self.mu_0).T @ self.sigma_inv
        self.thet_0 = 0.5*(self.mu_0 + self.mu_1).T @ self.sigma_inv @ (self.mu_0 - self.mu_1) - np.log((1-self.phi)/self.phi)


    def predict(self,X,prob=True):
        def gda_predict_single(X):
            return 1/(1 + np.exp(-(self.thet.T @ X + self.thet_0)))

        p = np.array(list(map(lambda x: gda_predict_single(x),[np.array(X)[i] for i in range(0,X.shape[0])])))
        if prob:
            return p
        else:
            return np.int64(p >= 0.5)
