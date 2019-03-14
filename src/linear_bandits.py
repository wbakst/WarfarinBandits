import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from utils import *

alpha = 7
NUM_RUNS = 5

original_data = pd.read_csv('data/warfarin.csv')
original_data = original_data[original_data['Therapeutic Dose of Warfarin'].notnull()]

# Train model parameters A and b
def run_UCB(alpha, K, d):
	# Randomize data for this run
	data = original_data.sample(frac=1).reset_index(drop=True)
	# Initialization
	num_correct = 0
	num_patients = 0
	avg_incorrect = []
	regret, error = 0.0, 0.0
	# Create (d,d) identity matrix
	A = np.array([np.identity(d), np.identity(d), np.identity(d)])
	# Initialize bias (d,1) b as all zeros
	b = np.array([np.zeros(d), np.zeros(d), np.zeros(d)])
	# Simulate Online Learning Environment
	for t, patient in data.iterrows():
		# Compute feature weights
		theta = [np.linalg.inv(Ai).dot(bi) for Ai, bi in zip(A, b)]
		# Observe (i.e. extract) K arms (in this case the same patient feature vector for each arm)
		try:
				X_t = np.array(get_linUCB_features(patient))
				num_patients += 1
		except:
				continue
		# Iterate over each arm
		p_t = np.zeros(K)
		for a in range(K):
				# compute probability distribution as theta_t.T.dot(X_t[a]) + alpha * sqrt(X_t[a].T.dot(Inverse(A)).dot(X_t[a]))
				p_t[a] = theta[a].T.dot(X_t) + alpha*np.sqrt(X_t.T.dot(np.linalg.inv(A[a])).dot(X_t)) # UCB (upper confidence bound)
		# Choose action a_t as argmax_a(p_t[a])
		a_t = np.argmax(p_t)
		# Observe reward r_t in {-1,0}
		true_action = get_true_action(patient)
		r_t = 1 if true_action == a_t else 0
		num_correct += 1 if true_action == a_t else 0
		avg_incorrect.append(((num_patients+1-num_correct) / (num_patients+1)))

		# Update A <-- A + x_t[a_t].dot(X_t[a_t].T)
		A[a_t] += X_t.dot(X_t.T)
		# Update b <-- b + x_t[a] * r_t
		b[a_t] += X_t*r_t
	# return regret and error
	print(num_correct / num_patients)
	return avg_incorrect, regret, error

for i in range(NUM_RUNS):
	avg_incorrect, regret, error = run_UCB(alpha, K=3, d=NUM_LIN_UCB_FEATURES)
	plt.plot(range(len(avg_incorrect)), avg_incorrect, label='run_{}'.format(i+1))
# plt.legend()
plt.show()
