import pandas as pd
import numpy as np
from utils import *

data = pd.read_csv('data/warfarin.csv')
data = data[data['Therapeutic Dose of Warfarin'].notnull()]

# Train model parameters A and b
def run_UCB(alpha, K, d):
	regret, error = 0.0, 0.0
	# Create (d,d) identity matrix
	A = np.array([np.identity(d), np.identity(d), np.identity(d)])
	# Initialize bias (d,1) b as all zeros
	b = np.array([np.zeros(d), np.zeros(d), np.zeros(d)])
	# Simulate Online Learning Environment
	for t, patient in data.iterrows():
		# Compute feature weights
		theta = [np.linalg.inv(Ai).dot(b) for Ai, bi in zip(A, b)]
		# Observe (i.e. extract) K arms (in this case the same patient feature vector for each arm)
		X_t = get_linUCB_features(patient)
		# Iterate over each arm
		p_t = np.zeros(K)
		for a in range(K):
			# compute probability distribution as theta_t.T.dot(X_t[a]) + alpha * sqrt(X_t[a].T.dot(Inverse(A)).dot(X_t[a]))
			p_t[a] = theta[a].T.dot(X_t) + alpha*np.sqrt(X_t.T.dot(np.linalg.inv(A[a])).dot(X_t)) # UCB (upper confidence bound)
		# Choose action a_t as argmax_a(p_t[a])
		a_t = np.argmax(p_t)
		# Observe reward r_t in {-1,0}
		r_t = 0
		# Update A <-- A + x_t[a_t].dot(X_t[a_t].T)
		A[a_t] += X_t.dot(X_t.T)
		# Update b <-- b + x_t[a] * r_t
		b[a_t] += X_t*r_t
	# return regret and error
	return regret, error

regret, error = run_UCB(alpha, K, d)
