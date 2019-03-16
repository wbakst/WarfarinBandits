from modules import MWU, ThompsonSampler, LinearUCB, LASSO
from utils import *

import sys
import argparse
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--algo',     type=str,    default='lin_ucb',    help='Algorithm to run on Wargarin Dataset')
parser.add_argument('--K',        type=int,    default=3,             help='Number of arms')
parser.add_argument('--d',        type=int,    default=NUM_FEATURES,  help='Number of features for each patient')

# UCB args
parser.add_argument('--alpha',    type=int,    default=7,             help='Alpha for LinearUCB')

# Thompson args
parser.add_argument('--epsilon',  type=float,  default=.5,            help='Epsilon for Thompson')
parser.add_argument('--delta',    type=float,  default=.5,            help='Delta for Thompson')
parser.add_argument('--R',        type=float,  default=1.,            help='R for Thompson')

# MWU args
parser.add_argument('--N',        type=int,    default=100,            help='How many experts to use for MWU')
parser.add_argument('--eta',      type=float,  default=0.95,          help='MWU exploration parameter')

args = parser.parse_args()

# Valid algorithms we have implements
BASELINES = ['single', 'lin_reg']
ALGORITHMS = ['mwu', 'thompson', 'lin_ucb', 'lasso']

# Read in data
data = pd.read_csv('data/warfarin.csv')
data = data[data['Therapeutic Dose of Warfarin'].notnull()]
data = data.sample(frac=1).reset_index(drop=True)

# Runs the baselines
def baseline():
	if args.algo == 'single':
		return single_action_baseline(data)
	elif args.algo == 'lin_reg':
		return linear_regression_baseline(data)
	else:
		raise NotImplementedError

# Runs a module from modules.py
def run():
	if args.algo == 'mwu':
		module = MWU(args.K, args.d, args.N, args.eta)
	elif args.algo == 'thompson':
		module = ThompsonSampler(args.K, args.d, args.epsilon, args.delta, args.R)
	elif args.algo == 'lin_ucb':
		module = LinearUCB(args.K, args.d, len(data))
	elif args.algo == 'lasso':
		module = LASSO(args.K, len(data), args.d, q=1, h=5, l1=0.05, l2=0.05)
	else:
		raise NotImplementedError

	# Maintain variables for later statistics
	num_correct = 0
	num_patients = 0
	avg_incorrect = []
	preds = [0, 0, 0]
	true = [0, 0, 0]

	# Iterate over patients
	for t, patient in data.iterrows():
		# Compute feature weights
		try:
			# skip tells us the data points that were skipped by linear regression
			# if we want to check accuracy on same data set as lin reg,
			# uncomment the line below
			X_t, skip = np.array(get_linUCB_features(patient))
			# if skip: continue
			num_patients += 1
		except Exception as e:
			print(e)
			continue

		# Pull an arm
		a_t = module.pull(X_t, t)
		preds[a_t] += 1

		# Observe reward r_t in {-1,0}
		true_action = get_true_action(patient)
		r_t = np.zeros(args.K)
		r_t[true_action] = 1
		true[true_action] += 1

		# Update the model
		module.update(X_t, a_t, r_t)

		# Update statistics variables
		num_correct += 1 if true_action == a_t else 0
		avg_incorrect.append(((t+1-num_correct) / (t+1)))

	# Return statisics variables
	return num_correct, num_patients, avg_incorrect, preds, true

def main():
	# If not MWU, then we can simply determine the
	# model and run it with the same skeleton
	num_correct, num_patients, avg_incorrect = 0, 0, 0
	if args.algo in BASELINES:
		num_correct, num_patients = baseline()
	elif args.algo in ALGORITHMS:
		num_correct, num_patients, avg_incorrect, preds, true = run()
		# Determine statistics and make plots
		# plt.plot(range(len(avg_incorrect)), avg_incorrect)
		# plt.show()
		# plt.close()
	else:
		raise NotImplementedError

	# Print out accuracy of algorithm
	print('Accuracy: {}'.format(num_correct / float(num_patients)))
	print('Predictions', preds)
	print('True', true)

if __name__ == '__main__':
	main()