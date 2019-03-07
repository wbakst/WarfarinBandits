import pandas as pd
import numpy as np

data = pd.read_csv('data/warfarin.csv')
data = data[data['Therapeutic Dose of Warfarin'].notnull()]


def train(alpha, K, d):
	# Create (d,d) identity matrix
	A = np.identity(d)
	# Initialize bias (d,1) b as all zeros
	b = np.zeros(d)
	# Simulate Online Learning Environment
	for t in range(T):
		# theta_t = Inverse(A).dot(b)

		# Observe (i.e. extract) K features (likely from patient data) called X_t

		for a in range(K):
			# compute probability distribution as theta_t.T.dot(X_t[a]) + alpha * sqrt(X_t[a].T.dot(Inverse(A)).dot(X_t[a]))

			# Above is equivalent to computing upper confidence bound

		# Choose action a_t as argmax_a(p_t[a])

		# Observe reward r_t in {0,1}

		# Update A <-- A + x_t[a_t].dot(X_t[a_t].T)

		# Update b <-- b + x_t[a] * r_t

	# return A and b






