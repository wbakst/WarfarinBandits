########################################
########## UTILITY FUNCTIONS ###########
########################################

LOW = 0
MEDIUM = 1
HIGH = 2

NUM_LIN_UCB_FEATURES = 2

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
	age = int(patient['Age'][0])
	# Extract heigh and weight
	height = patient['Height (cm)']
	weight = patient['Weight (kg)']
	# Extract race
	race = patient['Race']
	asian = 1 if race is 'Asian' else 0
	black = 1 if race is 'Black' else 0
	missing = 1 if race is 'Unknown' else 0
	# Extract medication information
	enzyme = get_enzyme_status(patient)
	amiodorone_status = get_amiodorone_status(patient)
	# Return feature vector of extract patient information
	return [1, age, height, weight, asian, black, missing, enzyme, amiodorone_status]

def get_height_features(height):
	if height < 120:
		return [1, 0, 0, 0]
	elif height < 160:
		return [0, 1, 0, 0]
	elif height < 180:
		return [0, 0, 1, 0]
	else:
		return [0, 0, 0, 1]


def get_age_features(age):
	features = [0 for i in range(10)]
	features[age] = 1
	return features


def get_weight_features(weight):
	if weight < 50:
		return [1, 0, 0, 0, 0]
	elif weight < 70:
		return [0, 1, 0, 0, 0]
	elif weight < 90:
		return [0, 0, 1, 0, 0]
	elif weight < 110:
		return [0, 0, 0, 1, 0]
	return [0, 0, 0, 0, 1]

def get_CYP2C9_features(CYP2C9):
	if '*1/*2' in CYP2C9: return [1, 0, 0, 0, 0, 0]
	elif '*1/*3' in CYP2C9: return [0, 1, 0, 0, 0, 0]
	elif '*2/*2' in CYP2C9: return [0, 0, 1, 0, 0, 0]
	elif '*2/*3' in CYP2C9: return [0, 0, 0, 1, 0, 0]
	elif '*3/*3' in CYP2C9: return [0, 0, 0, 0, 1, 0]
	else: return [0, 0, 0, 0, 0, 1]


def get_VKORC1_features(VKORC1):
	if 'A/A' in VKORC1: return [1, 0, 0]
	elif 'A/G' in VKORC1: return [0, 1, 0]
	else: return [0, 0, 1]

# Extract (linUCB) feature vector for a given patient
def get_linUCB_features(patient):

	features = get_baseline_linear_features(patient)


	age = features[1]
	height = features[2]
	weight = features[3]

	del features[1:7]
	del features[1]

	# features += get_age_features(age)
	# features += get_height_features(height)
	# features += get_weight_features(weight)
	# features += get_CYP2C9_features(patient['Cyp2C9 genotypes'])
	# # print (patient['VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T'])
	# #print(patient['VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T'])
	# features += get_VKORC1_features(patient['VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T'])

	return features

########################################
###### INTERNAL HELPER FUNCTIONS #######
########################################

def get_amiodorone_status(patient):
	medications = patient['Medications']
	if not isinstance(medications, str) and np.isnan(medications):
		return 0
	if 'amiodarone' in medications and 'not amiodarone' not in medications:
		return 1
	return 0

def get_enzyme_status(patient):
	num_inducers = patient['Rifampin or Rifampicin'] + patient['Carbamazepine (Tegretol)'] + patient['Phenytoin (Dilantin)']
	return 1 if num_inducers > 0 else 0
