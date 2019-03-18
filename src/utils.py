########################################
########## UTILITY FUNCTIONS ###########
########################################
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import metrics
from matplotlib import pyplot as plt

LOW = 0
MEDIUM = 1
HIGH = 2

NUM_LIN_UCB_FEATURES = 65
NUM_FEATURES = 62

data_cols = ['PharmGKB Subject ID', 'Gender', 'Race', 'Ethnicity', 'Age', 'Height (cm)', 'Weight (kg)',
'Indication for Warfarin Treatment', 'Comorbidities', 'Diabetes', 'Congestive Heart Failure and/or Cardiomyopathy',
'Valve Replacement', 'Medications', 'Aspirin', 'Acetaminophen or Paracetamol (Tylenol)',
'Was Dose of Acetaminophen or Paracetamol (Tylenol) >1300mg/day', 'Simvastatin (Zocor)',
'Atorvastatin (Lipitor)', 'Fluvastatin (Lescol)', 'Lovastatin (Mevacor)', 'Pravastatin (Pravachol)',
'Rosuvastatin (Crestor)', 'Cerivastatin (Baycol)', 'Amiodarone (Cordarone)', 'Carbamazepine (Tegretol)',
'Phenytoin (Dilantin)', 'Rifampin or Rifampicin', 'Sulfonamide Antibiotics', 'Macrolide Antibiotics',
'Anti-fungal Azoles', 'Herbal Medications, Vitamins, Supplements', 'Target INR',
'Estimated Target INR Range Based on Indication', 'Subject Reached Stable Dose of Warfarin',
'Therapeutic Dose of Warfarin', 'INR on Reported Therapeutic Dose of Warfarin', 'Current Smoker',
'Cyp2C9 genotypes', 'Genotyped QC Cyp2C9*2', 'Genotyped QC Cyp2C9*3', 'Combined QC CYP2C9',
'VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T', 'VKORC1 QC genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T',
'VKORC1 genotype: 497T>G (5808); chr16:31013055; rs2884737; A/C', 'VKORC1 QC genotype: 497T>G (5808); chr16:31013055; rs2884737; A/C',
'VKORC1 genotype: 1173 C>T(6484); chr16:31012379; rs9934438; A/G', 'VKORC1 QC genotype: 1173 C>T(6484); chr16:31012379; rs9934438; A/G',
'VKORC1 genotype: 1542G>C (6853); chr16:31012010; rs8050894; C/G', 'VKORC1 QC genotype: 1542G>C (6853); chr16:31012010; rs8050894; C/G',
'VKORC1 genotype: 3730 G>A (9041); chr16:31009822; rs7294;  A/G', 'VKORC1 QC genotype: 3730 G>A (9041); chr16:31009822; rs7294;  A/G',
'VKORC1 genotype: 2255C>T (7566); chr16:31011297; rs2359612; A/G', 'VKORC1 QC genotype: 2255C>T (7566); chr16:31011297; rs2359612; A/G',
'VKORC1 genotype: -4451 C>A (861); Chr16:31018002; rs17880887; A/C', 'VKORC1 QC genotype: -4451 C>A (861); Chr16:31018002; rs17880887; A/C',
'CYP2C9 consensus', 'VKORC1 -1639 consensus', 'VKORC1 497 consensus', 'VKORC1 1173 consensus', 'VKORC1 1542 consensus', 'VKORC1 3730 consensus',
'VKORC1 2255 consensus', 'VKORC1 -4451 consensus', 'Unnamed: 63', 'Unnamed: 64', 'Unnamed: 65']

#  linear_features = {
# 	 'bias': 0,										# Possible values: N/A
# 	 'age': 1,										# Possible values: N/A
# 	 'height': 2,									# Possible values: N/A
# 	 'weight': 3,									# Possible values: N/A
# 	 'race_asian': 4,							# Possible values: N/A
# 	 'race_black': 5,							# Possible values: N/A
# 	 'race_missing_or_mixed': 6,	# Possible values: N/A
# 	 'enzyme': 7,									# Possible values: N/A
# 	 'amiadorone': 8,							# Possible values: N/A
#  }
baseline_feature_weights = {
  'bias': 4.0375,
  'Age:': -0.2546,
  'Height (cm)': 0.0118,
  'Weight (kg)': 0.0134,
  'Asian_Race': -0.6752,
  'Black_Race': 0.4060,
  'Missing_Race': 0.0443,
  'Enzyme': 1.2799,
  'Amiodarone': -0.5695
}

baseline_features = [
  'bias',
  'Age:',
  'Height (cm)',
  'Weight (kg)',
  'Asian_Race',
  'Black_Race',
  'Missing_Race',
  'Enzyme',
  'Amiodarone'
]

# simple plotting function
def plot_stats(stat, title):
	plt.plot(range(len(stat)), stat)
	plt.title(title)
	plt.xlabel('Timestep')
	plt.show()

 # Returns which dosage type (low, medium, high)
def get_dosage_bucket(dosage):
	if dosage < 21:
		return LOW
	elif dosage <= 49:
		return MEDIUM
	else:
		return HIGH

# Determines if predicted dosage matches the true dosage from the data
def correct_predicted_dosage(true_dosage, predicted_dosage):
	return get_dosage_bucket(true_dosage) == get_dosage_bucket(predicted_dosage)

# Extract true action (LOW, MEDIUM, HIGH) for a given patient
def get_true_action(patient):
	return get_dosage_bucket(float(patient['Therapeutic Dose of Warfarin']))

# Extract true dosage for a given patient
def get_true_dosage(patient):
	return float(patient['Therapeutic Dose of Warfarin'])

# Extract (baseline linear regression) feature vector for a given patient
'''
for index, patient in data.iterrows():
  try:
    f = get_baseline_linear_features(patient)
    num_patients += 1
  except:
    continue # skip rows with missing entries
'''
def get_baseline_linear_features(patient):
	# Extract age (in decades)
	age = patient['Age']

	if not pd.isnull(age):
		age = int(patient['Age'][0])

	# Extract heigh and weight
	height = patient['Height (cm)']
	weight = patient['Weight (kg)']

	# Extract race
	race = patient['Race']

	asian = 1 if 'Asian' in race else 0
	black = 1 if 'Black' in race else 0
	missing = 1 if 'Unknown' in race else 0
	# Extract medication information
	enzyme = get_enzyme_status(patient)
	amiodorone_status = get_amiodorone_status(patient)
	# Return feature vector of extract patient information
	return [1, age, height, weight, asian, black, missing, enzyme, amiodorone_status]

def get_height_features(height):
	if np.isnan(height):
		return [1, 0, 0, 0, 0]
	if height < 165:
		return [0, 1, 0, 0, 0]
	elif height < 177:
		return [0, 0, 1, 0, 0]
	elif height < 190:
		return [0, 0, 0, 1, 0]
	else:
		return [0, 0, 0, 0, 1]

def get_age_features(age):
	features = [0 for i in range(10)]
	if np.isnan(age):
		features[0] = 1
		return features
	# We bucket 90+ into 80s
	elif age > 8: age = 8
	features[age+1] = 1
	return features


def get_weight_features(weight):
	if np.isnan(weight):
		return [1, 0, 0, 0, 0, 0, 0, 0]
	if weight < 46:
		return [0, 1, 0, 0, 0, 0, 0, 0]
	elif weight < 60:
		return [0, 0, 1, 0, 0, 0, 0, 0]
	elif weight < 72:
		return [0, 0, 0, 1, 0, 0, 0, 0]
	elif weight < 86:
		return [0, 0, 0, 0, 1, 0, 0, 0]
	elif weight < 100:
		return [0, 0, 0, 0, 0, 1, 0, 0]
	elif weight < 113:
		return [0, 0, 0, 0, 0, 0, 1, 0]
	return [0, 0, 0, 0, 0, 0, 0, 1]

def get_CYP2C9_features(CYP2C9):
	if pd.isnull(CYP2C9):features = [1, 0, 0, 0, 0, 0, 0]
	elif '*1/*2' in CYP2C9: features = [0, 1, 0, 0, 0, 0, 0]
	elif '*1/*3' in CYP2C9: features = [0, 0, 1, 0, 0, 0, 0]
	elif '*2/*2' in CYP2C9: features = [0, 0, 0, 1, 0, 0, 0]
	elif '*2/*3' in CYP2C9: features = [0, 0, 0, 0, 1, 0, 0]
	elif '*3/*3' in CYP2C9: features = [0, 0, 0, 0, 0, 1, 0]
	else: features = [0, 0, 0, 0, 0, 0, 1]
	return features

def imputeVKORC1(patient):
	race = patient['Race']
	# if 'Black' not in race
	rs2359612 = patient['VKORC1 genotype: 2255C>T (7566); chr16:31011297; rs2359612; A/G']
	rs9934438 =	patient['VKORC1 genotype: 1173 C>T(6484); chr16:31012379; rs9934438; A/G']
	rs8050894	= patient['VKORC1 genotype: 1542G>C (6853); chr16:31012010; rs8050894; C/G']
	if 'Black' not in race and 'Unknown' not in race and not pd.isnull(rs2359612) and 'C/C' in rs2359612:
		patient['VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T'] = 'G/G'
	elif 'Black' not in race and 'Unknown' not in race and not pd.isnull(rs2359612) and 'T/T' in rs2359612:
		patient['VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T'] = 'A/A'
	elif 'Black' not in race and 'Unknown' not in race and not pd.isnull(rs2359612) and 'C/T' in rs2359612:
		patient['VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T'] = 'A/G'
	elif not pd.isnull(rs9934438) and 'C/C' in rs9934438:
		patient['VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T'] = 'G/G'
	elif not pd.isnull(rs9934438) and 'T/T' in rs9934438:
		patient['VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T'] = 'A/A'
	elif not pd.isnull(rs9934438) and 'C/T' in rs9934438:
		patient['VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T'] = 'A/G'
	elif 'Black' not in race and 'Unknown' not in race and not pd.isnull(rs8050894) and 'G/G' in rs8050894:
		patient['VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T'] = 'G/G'
	elif 'Black' not in race and 'Unknown' not in race and not pd.isnull(rs8050894) and 'C/C' in rs8050894:
		patient['VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T'] = 'A/A'
	elif 'Black' not in race and 'Unknown' not in race and not pd.isnull(rs8050894) and 'C/G' in rs8050894:
		patient['VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T'] = 'A/G'
	return patient['VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T']

def get_VKORC1_features(VKORC1, patient):
	if pd.isnull(VKORC1): VKORC1 = imputeVKORC1(patient)
	if pd.isnull(VKORC1):	features  = [1, 0, 0, 0]
	elif 'A/A' in VKORC1: features = [0, 1, 0, 0]
	elif 'A/G' in VKORC1: features = [0, 0, 1, 0]
	else:	features = [0, 0, 0, 1]
	return features

# 0 -- no disease
# 1 -- has disease
def get_comorbidities(comorbidities):
	status = 0
	if not pd.isnull(comorbidities):
		comorbidities = comorbidities.lower()
		if 'no comorbidities' in comorbidities or 'none' in comorbidities or 'no cancer' in comorbidities:
			status = 0
		else:
			status = 1
	return [status]

# 9 features, 0 index is missing
def get_indication_for_treatment(indication):
	features = [0 for i in range(9)]
	if pd.isnull(indication):
		features[0] = 1
	elif 'or' in indication:
		indices = [int(i.strip()) for i in indication.split('or')]
		for idx in indices:
			features[idx] = 1
	elif ';' in indication:
		indices = [int(i.strip()) for i in indication.split(';')]
		for idx in indices:
			features[idx] = 1
	elif ',' in indication:
		indices = [int(i.strip()) for i in indication.split(',')]
		for idx in indices:
			features[idx] = 1
	else:
		features[int(indication)] = 1
	return features

# 0 missing
# 1 no
# 2 yes
def get_binary_feature(feature):
	features = [0, 0, 0]
	if pd.isnull(feature): features[0] += 1
	elif int(feature) == 0: features[1] += 1
	elif int(feature) == 1: features[2] += 1
	else: print('uh oh! should not get here')
	return features

def get_diabetes(patient):
	return get_binary_feature(patient['Diabetes'])

def get_smoker(patient):
	return get_binary_feature(patient['Current Smoker'])

def get_simvastatin(patient):
	return get_binary_feature(patient['Simvastatin (Zocor)'])
def get_atorvastatin(patient):
	return get_binary_feature(patient['Atorvastatin (Lipitor)'])
def get_fluvastatin(patient):
	return get_binary_feature(patient['Fluvastatin (Lescol)'])
def get_lovastatin(patient):
	return get_binary_feature(patient['Lovastatin (Mevacor)'])
def get_pravastatin(patient):
	return get_binary_feature(patient['Pravastatin (Pravachol)'])
def get_rosuvastatin(patient):
	return get_binary_feature(patient['Rosuvastatin (Crestor)'])
def get_cerivastatin(patient):
	return get_binary_feature(patient['Cerivastatin (Baycol)'])

# Extract feature vector for a given patient
def get_features(patient, ucb=False):
	features = get_baseline_linear_features(patient)
	skip = False

	age = features[1]
	height = features[2]
	weight = features[3]
	if pd.isnull(age) or pd.isnull(height) or pd.isnull(weight):
		skip = True

	# del features[1:7]
	del features[1:4]

	features += get_age_features(age)
	features += get_height_features(height)
	features += get_weight_features(weight)
	features += get_CYP2C9_features(patient['Cyp2C9 genotypes'])
	features += get_VKORC1_features(patient['VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T'], patient)
	features += get_comorbidities(patient['Comorbidities'])
	features += get_indication_for_treatment(patient['Indication for Warfarin Treatment'])
	features += get_diabetes(patient)
	# features += get_smoker(patient)
	features += get_simvastatin(patient)
	features += get_atorvastatin(patient)
	features += get_fluvastatin(patient)
	# features += get_gender(patient)
	# features += get_lovastatin(patient)
	# features += get_pravastatin(patient)
	#features += get_rosuvastatin(patient)
	# features += get_cerivastatin(patient)
	if ucb:
		features = [0, 0, 0] + features
		return np.reshape(preprocessing.normalize(np.reshape(features, (1, -1))), (NUM_LIN_UCB_FEATURES,)), skip
	else:
		return np.reshape(preprocessing.normalize(np.reshape(features, (1, -1))), (NUM_FEATURES,)), skip

########################################
######## Metrics ########
########################################
def get_confusion_matrix(predictions, true, K, data, percentage):
	cm = metrics.confusion_matrix(true, predictions).astype(float)
	if percentage:
		num_low = sum([1 for dose in true if dose == 0])
		num_med =  sum([1 for dose in true if dose == 1])
		num_high = sum([1 for dose in true if dose == 2])
		cm[0] = cm[0] / num_low
		cm[1] = cm[1] / num_med
		cm[2] = cm[2] / num_high
		cm = cm * 100

	return cm

########################################
###### INTERNAL HELPER FUNCTIONS #######
########################################

def get_amiodorone_status(patient):
	medications = patient['Medications']
	status = 0
	if not pd.isnull(medications): medications = medications.lower()
	if not pd.isnull(medications) and 'amiodarone' in medications and 'not amiodarone' not in medications and 'no amiodarone' not in medications:
		status = 1
	return status

def get_enzyme_status(patient):
	inducer_1 = patient['Rifampin or Rifampicin']
	inducer_2 = patient['Carbamazepine (Tegretol)']
	inducer_3 = patient['Phenytoin (Dilantin)']
	if pd.isnull(inducer_1) or pd.isnull(inducer_2) or pd.isnull(inducer_3): return 0
	num_inducers = inducer_1 + inducer_2 + inducer_3
	return 1 if num_inducers > 0 else 0

########################################
######## BASELINE FUNCTIONALITY ########
########################################

def single_action_baseline(data):
	true_dosages = data.loc[:,'Therapeutic Dose of Warfarin'].values.tolist()
	num_incorrect, num_correct = 0., 0
	num_patients = 0
	regret = []
	avg_incorrect = []

	for true_dosage in true_dosages:
		num_patients += 1
		if true_dosage >= 21 and true_dosage <=49: num_correct += 1
		else: num_incorrect += 1
		avg_incorrect.append(num_incorrect / num_patients)
		regret.append(num_incorrect)

	return num_correct, len(true_dosages), avg_incorrect, [MEDIUM for i in range(len(true_dosages))], true_dosages, regret

def linear_regression_baseline(data):
	num_correct, num_patients, num_incorrect, num_discarded = 0, 0, 0, 0
	preds = []
	true = []
	regret = []
	avg_incorrect = []
	for index, patient in data.iterrows():
		f = get_baseline_linear_features(patient)
		if True in np.isnan(f):
			num_discarded += 1
			continue # Skip rows with missing data
		num_patients += 1
		pred = 0
		for index, item in enumerate(f):
			pred += item * baseline_feature_weights[baseline_features[index]]
		pred = pred * pred
		preds.append(get_dosage_bucket(pred))
		true_dosage = get_true_dosage(patient)
		true.append(get_dosage_bucket(true_dosage))
		if correct_predicted_dosage(true_dosage, pred):
			num_correct += 1
		else:
			num_incorrect += 1
		regret.append(num_incorrect)
		avg_incorrect.append(float(num_incorrect) / num_patients)

	return num_correct, float(num_patients), avg_incorrect, preds, true, regret
