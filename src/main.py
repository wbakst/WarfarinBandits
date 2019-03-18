from modules import MWU, ThompsonSampler, LinearUCB, LASSO
from utils import *
from utils import get_confusion_matrix

import sys
import argparse
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--algo',         type=str,    default='lin_ucb',     help='Algorithm to run on Wargarin Dataset')
parser.add_argument('--K',            type=int,    default=3,             help='Number of arms')
parser.add_argument('--d',            type=int,    default=NUM_FEATURES,  help='Number of features for each patient')

# UCB args
parser.add_argument('--alpha',        type=int,    default=7,             help='Alpha for LinearUCB')

# LASSO args
parser.add_argument('--q',            type=int,    default=3,             help='q value for lasso')
parser.add_argument('--h',            type=float,  default=1.,            help='h value for lasso')
parser.add_argument('--n',            type=int,    default=5,             help='n value for lasso')
parser.add_argument('--l1',           type=float,  default=0.05,          help='lambda_1 value for lasso')
parser.add_argument('--l2',           type=float,  default=0.05,          help='lambda_2 value for lasso')

# Thompson args
# parser.add_argument('--epsilon',  type=float,  default=.5,            help='Epsilon for Thompson')
# parser.add_argument('--delta',    type=float,  default=.5,            help='Delta for Thompson')
# parser.add_argument('--R',        type=float,  default=1.,            help='R for Thompson')
parser.add_argument('--v',            type=float,  default=.25,           help='v for Thompson')

# MWU args
parser.add_argument('--N',            type=int,    default=10,            help='How many experts to use for MWU')
parser.add_argument('--eta',          type=float,  default=0.95,          help='MWU exploration parameter')
parser.add_argument('--expert_type',  type=str,    default='thompson',    help='Which module to use as an expert for MWU')

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
	if args.algo in BASELINES:
		return baseline()
	if args.algo == 'mwu':
		module = MWU(args.K, args.d, args.N, args.eta, args.l2, args.expert_type)
	elif args.algo == 'thompson':
		module = ThompsonSampler(args.K, args.d, args.v)
	elif args.algo == 'lin_ucb':
		args.d = NUM_LIN_UCB_FEATURES
		module = LinearUCB(args.K, args.d, len(data))
	elif args.algo == 'lasso':
		# module = LASSO(args.K, len(data), args.d, q=args.q, h=args.h, l1=args.l1, l2=args.l2)
		module = LASSO(args.K, args.d, args.h, args.q, args.n, args.l1, args.l2)
	else:
		raise NotImplementedError

	# Maintain variables for later statistics
	num_correct = 0
	num_patients = 0
	avg_incorrect = []
	preds = []
	true = []
	num_incorrect = 0
	regret = []

	# Iterate over patients
	for t, patient in data.iterrows():
		# Compute feature weights
		try:
			# skip tells us the data points that were skipped by linear regression
			# if we want to check accuracy on same data set as lin reg,
			# uncomment the line below
			X_t, skip = np.array(get_features(patient, (args.algo == 'lin_ucb')))
			# if skip: continue
		except Exception as e:
			print(e)
			continue

		num_patients += 1
		# Pull an arm
		a_t = module.pull(X_t)
		preds.append(a_t)

		# Observe reward r_t in {-1,0}
		true_action = get_true_action(patient)
		r_t = np.zeros(args.K)
		r_t[true_action] = 1
		true.append(true_action)

		# Update the model
		module.update(X_t, a_t, r_t)

		# Update statistics variables
		if true_action == a_t: num_correct += 1
		else:	num_incorrect += 1

		regret.append(num_incorrect)
		avg_incorrect.append(num_incorrect / num_patients)

	# Return statisics variables
	return num_correct, num_patients, avg_incorrect, preds, true, regret

def main():
	num_correct, num_patients, avg_incorrect = 0, 0, 0
	num_correct, num_patients, avg_incorrect, preds, true, regret = run()
	print('Accuracy: {}'.format(num_correct / float(num_patients)))
	plot_stats(avg_incorrect, 'average incorrect')
	plot_stats(regret, 'regret')


if __name__ == '__main__':
	main()