from scipy.stats import multivariate_normal
import numpy as np
from utils import *

##################################################
################# LASSO BANDIT ###################
##################################################

class LASSO:
	def __init__(self, K, n, d, q, h, l1, l2):
		# Maintain list of actions and rewards for future reference
		self.actions, self.rewards = [], []
		# Parameters
		self.K = K
		# TODO: change this
		self.n = 10
		self.d = d
		self.q = 3
		self.h = 2 # What should h really be here?
		self.l1 = l1
		self.l2 = l2
		# Variables
		# Forced sample set
		self.T = [self.construct_forced_sample_set(i+1) for i in range(self.K)]
		print(self.T)
		# All sample set
		self.S = [set() for i in range(self.K)]
		# Initialize Beta_T, Beta_S as d dimensional vectors
		self.Beta_T = [np.zeros(d) for i in range(K)]
		self.Beta_S = [np.zeros(d) for i in range(K)]

	# Ratchet
	def construct_forced_sample_set(self, i):
		samples = []
		j_vals = []
		k = 1
		val = 0
		while val < self.q*i:
			val = self.q*(i-1) + k
			j_vals.append(val)
			k += 1
		for n_i in range(self.n):
			for j in j_vals:
				samples.append((2**n_i)*self.K*self.q+j)
		return set(samples)


	# Need to add t to all pulls
	def pull(self, X_t, t):
		# If t in any of forced sample sets, return action
		for i in self.K:
			if t in self.T[i]: return i
		# If t is not in any of the forced sample sets:
		#	We used the forced sample estimates Beta_T to find a subset of actions that maximize reward 1
		# We then use the all sample estimates to choose the arm with the highest estimated reward
		# within the subset of actions


	def update(self, X_t, a_t, r_t):
		pass

##################################################
################### LINEAR UCB ###################
##################################################

class LinearUCB:
	def __init__(self, K, d, n):
		# Maintain list of actions and rewards for future reference
		self.actions, self.rewards = [], []
		# Parameters
		self.K = K
		self.d = d
		self.n = n
		# New alpha
		self.alpha = 0.5*np.log(2*self.n*self.K/(0.1))
		# Variables
		self.A = np.array([np.identity(d) for i in range(K)])
		self.b = np.array([np.zeros(d) for i in range(K)])

	def pull(self, X_t, t):
		# Compute feature weights
		theta = [np.linalg.inv(Ai).dot(bi) for Ai, bi in zip(self.A, self.b)]
		# Iterate over each arm
		p_t = np.zeros(self.K)
		for a in range(self.K):
			# compute probability distribution as theta_t.T.dot(X_t[a]) + alpha * sqrt(X_t[a].T.dot(Inverse(A)).dot(X_t[a]))
			p_t[a] = theta[a].T.dot(X_t) + self.alpha*np.sqrt(X_t.T.dot(np.linalg.inv(self.A[a])).dot(X_t)) # UCB (upper confidence bound)
		# Choose action a_t as argmax_a(p_t[a])
		a_t = np.argmax(p_t)
		# Update actions list and return chosen arm
		self.actions.append(a_t)
		return a_t

	def update(self, X_t, a_t, r_t):
		# Update A <-- A + x_t[a_t].dot(X_t[a_t].T)
<<<<<<< Updated upstream
		self.A[a_t] += np.outer(X_t, X_t)
=======
		self.A[a_t] += np.outer(X_t, X_t.T)
>>>>>>> Stashed changes
		# Update b <-- b + x_t[a] * r_t
		self.b[a_t] += X_t*r_t[a_t]

##################################################
############### THOMPSON SAMPLING ################
##################################################
class ThompsonSampler:
	def __init__(self, K, d, v=.25):
		# Maintain list of actions and rewards for future reference
		self.actions, self.rewards = [], []
		# Parameters
		self.K = K
		self.d = d
		# Hyperparameters
		# self.epsilon = epsilon
		# self.delta = delta
		# self.R = R
		# Variables
		self.B = [np.identity(d) for i in range(K)]
		self.mu_hat = [np.zeros(d) for i in range(K)]
		self.f = [np.zeros(d) for i in range(K)]
		# self.v = self.R * np.sqrt(24./self.epsilon * self.d * np.log(1./self.delta))
		self.v = v

	def pull(self, X_t, t):
		# Iterate over actions
		b = np.zeros(self.K)
		for a in range(self.K):
			# sample mu_tilde(t) from Normal(mu_hat, v^2.dot(inverse(B)))
			mu_tilde = multivariate_normal.rvs(self.mu_hat[a], self.v**2 * np.linalg.inv(self.B[a]))
			b[a] = X_t.T.dot(mu_tilde)
		# a(t) = argmax_i b_i(t)^T mu_tilde(t)
		a_t = np.argmax(b)
		# Update actions list and return chosen arm
		self.actions.append(a_t)
		return a_t

	def update(self, X_t, a_t, r_t):
		# Update rewards
		self.rewards.append(r_t[a_t])
		# Update parameters based on choice and reward
		self.B[a_t] = self.B[a_t] + np.outer(X_t, X_t)
		self.f[a_t] = self.f[a_t] + X_t * r_t[a_t]
		self.mu_hat[a_t] = np.linalg.inv(self.B[a_t]).dot(self.f[a_t])

##################################################
########## MULTIPLICATIVE WEIGHT UPDATE ##########
##################################################
class MWU:
	def __init__(self, K, d, N, eta):
		# Maintain list of actions and rewards for future reference
		self.actions, self.rewards = [], []
		# Parameters
		self.N = N
		self.eta = eta
		self.K = K
		self.d = d
		# Experts
		self.experts = [ThompsonSampler(self.K, self.d) for i in range(self.N)]
		# Variables
		self.weights = [1.] * self.N
		self.previous_expert_actions = [0] * len(self.experts)

	def pull(self, X_t, t):
		# Pull an arm for each expert
		weighted_actions = [0.] * self.K
		for i in range(self.N):
			a = self.experts[i].pull(X_t, t)
			weighted_actions[a] += self.weights[i] / np.sum(self.weights)
			self.previous_expert_actions[i] = a
		# Determine majority vote action
		a_t = np.argmax(weighted_actions)
		# Update actions list and return chosen arm
		self.actions.append(a_t)
		return a_t

	def update(self, X_t, a_t, r_t):
		# Updae rewards
		self.rewards.append(r_t[a_t])
		# Update each individual expert
		for i in range(self.N):
			self.experts[i].update(X_t, a_t, r_t)
		# Update weights based on correct vs. incorrect guesses
		for i, weight in enumerate(self.weights.copy()):
			self.weights[i] *= (self.eta ** -r_t[self.previous_expert_actions[i]])
