import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from problem_1 import LogisticRegression, confusion_mat, plot_model

df_train = pd.read_csv("data/ds3_train.csv")
df_valid = pd.read_csv("data/ds3_valid.csv")
df_test = pd.read_csv("data/ds3_test.csv")

LR_true = LogisticRegression()
LR_true.train(df_train[["x_1","x_2"]],df_train["t"])

LR_true_test = LR_true.predict(df_test[["x_1","x_2"]],prob=False)

print("LR trained with access to true labels")
print(confusion_mat(df_test["y"],LR_true_test))
# -----------------------

LR_posonly = LogisticRegression()
LR_posonly.train(df_train[["x_1","x_2"]],df_train["y"])
LR_posonly_test_noscale = LR_posonly.predict(df_test[["x_1","x_2"]],prob=False)


LR_valid_preds = LR_posonly.predict(df_valid[["x_1","x_2"]],prob=True)

alpha_estimate = sum(LR_valid_preds)/len(LR_valid_preds)

LR_posonly_rescaled_test_prob = (1/alpha_estimate)*LR_posonly.predict(df_test[["x_1","x_2"]],prob=True)
LR_posonly_rescaled_test_class = np.array([int(x >= 0.5) for x in LR_posonly_rescaled_test_prob])

print("LR trained with access to true labels")
print(confusion_mat(df_test["y"],LR_true_test))

print("LR trained with access to partial positive labels, no rescale")
print(confusion_mat(df_test["y"],LR_posonly_test_noscale))

print("LR trained with access to partial positive labels")
print(confusion_mat(df_test["y"],LR_posonly_rescaled_test_class))

# ------------------------
def plot_model(train,theta,title="",alpha_correct=0):
    fig,ax = plt.subplots()
    pos = train[train["t"]==1]
    neg = train[train["t"]==0]
    ax.scatter(pos["x_1"],pos["x_2"],marker='o')
    ax.scatter(neg["x_1"],neg["x_2"],marker='x')

    boundary_x = np.linspace(min(train["x_1"]),max(train["x_1"]))
    ax.plot(boundary_x,(alpha_correct + theta[0] + theta[1]*boundary_x)/(-theta[2]))
    fig.suptitle(title)
    return fig

#plot_model(df_valid,LR_true.theta,title="LR trained with true labels")
#plot_model(df_valid,LR_posonly.theta,title="LR with positive only labels, without alpha correction")
#plot_model(df_valid,LR_posonly.theta,alpha_correct=np.log(1/(alpha_estimate*0.5) - 1),title="LR with positive only labels, with alpha correction")
#

plot_model(df_test,LR_true.theta,title="LR trained with true labels")
plot_model(df_test,LR_posonly.theta,title="LR with positive only labels, without alpha correction")
plot_model(df_test,LR_posonly.theta,alpha_correct=np.log(2/(alpha_estimate) - 1),title="LR with positive only labels, with alpha correction")
