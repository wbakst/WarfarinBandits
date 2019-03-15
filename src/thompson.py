import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
import numpy as np
from utils import *

data = pd.read_csv('data/warfarin.csv')
data = data[data['Therapeutic Dose of Warfarin'].notnull()]


# Train model parameters A and b
def run_thompson(K, d):
    num_correct = 0
    num_patients = 0
    avg_incorrect = []

    regret, error = 0.0, 0.0

    B = [np.identity(d)] * K
    mu_hat = [np.zeros(d)] * K
    f = [np.zeros(d)] * K

    epsilon = .5
    delta = .5
    R = 1

    v = R * np.sqrt(24./epsilon * d * np.log(1./delta))

    for t, patient in data.iterrows():
        # Compute feature weights
        try:
            X_t = np.array(get_linUCB_features(patient))
            
            num_patients += 1
        except:
            continue
        
        # Iterate over actions
        b = np.zeros(K)
        for a in range(K):

            # sample mu_tilde(t) from Normal(mu_hat, v^2.dot(inverse(B)))
            mu_tilde = multivariate_normal.rvs(mu_hat[a], v**2 * np.linalg.inv(B[a]))

            b[a] = X_t.T.dot(mu_tilde)            

        # a(t) = argmax_i b_i(t)^T mu_tilde(t)
        a_t = np.argmax(b)

        # Observe reward r_t in {-1,0}
        true_action = get_true_action(patient)
        r_t = 1 if true_action == a_t else 0

        B[a_t] = B[a_t] + X_t.dot(X_t.T)
        f[a_t] = f[a_t] + X_t * r_t
        mu_hat[a_t] = np.linalg.inv(B[a_t]).dot(f[a_t])

        num_correct += 1 if true_action == a_t else 0
        avg_incorrect.append(((t+1-num_correct) / (t+1)))


    # return regret and error
    print(num_correct / float(num_patients))
    plt.plot(range(len(avg_incorrect)), avg_incorrect)
    plt.show()
    plt.close()
    return regret, error

run_thompson(K=3, d=25)
