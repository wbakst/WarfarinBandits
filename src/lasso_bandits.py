import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from utils import *

NUM_RUNS = 5

original_data = pd.read_csv('data/warfarin.csv')
original_data = original_data[original_data['Therapeutic Dose of Warfarin'].notnull()]

def run_LASSO(q, h, lam_1, lam_2_0, d):
	# Initialize T_i_0 and S_i_0 as empty sets
	T, S = [set(), set(), set()], [set(), set(), set()]
	# Initialize Beta_T, Beta_S as d dimensional vectors
	Beta_T, Beta_S = [np.zeros(d), np.zeros(d), np.zeros(d)], [np.zeros(d), np.zeros(d), np.zeros(d)]
	# 