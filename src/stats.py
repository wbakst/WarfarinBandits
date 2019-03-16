def get_statistics(filename):
	with open(filename) as file:
		high, low, total, count = float('-inft'), float('inf'), 0, 0
		for line in file:
			accuracy += float(line.strip())
			if accuracy > high: high = accuracy
			if accuracy < low: low = accuracy
			total += accuracy
			count += 1
		return total/count, high, low

FILES = [('LinUCB', 'data/lin_ucb_stats.txt'),
		 ('Thompson', 'data/thompson_stats.txt'),
		 ('LASSO', 'data/lasso_stats.txt'),
		 ('MWU (Thompson)', 'data/mwu_thompson_stats.txt'),
		 ('MWU (LASSO)', 'data/mwu_lasso_stats.txt')]

def main():
	for algo, filename in FILES:
		mean, high, low = get_statistics(filename)
		print('{} Accuracy: {} (+/-) {}'.format(algo, mean, min(high-mean, mean-low)))

if __name__ == '__main__':
	main()