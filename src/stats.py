import numpy as np

def get_statistics(filename):
	with open(filename) as file:
		accuracies = [float(line.strip().split()[1]) for line in file]
		return np.mean(accuracies), np.std(accuracies)

FILES = [('LinUCB', 'stats/lin_ucb_stats.txt'),
		 ('Thompson', 'stats/thompson_stats.txt'),
		 ('LASSO', 'stats/lasso_stats.txt'),
		 ('MWU (Thompson)', 'stats/mwu_thompson_stats.txt'),
		 ('MWU (LASSO)', 'stats/mwu_lasso_stats.txt')]

def main():
	for algo, filename in FILES:
		mean, std = get_statistics(filename)
		print('{} Accuracy: {} (+/-) {}'.format(algo, mean, std))

if __name__ == '__main__':
	main()
