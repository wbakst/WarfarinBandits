########################################
########## UTILITY FUNCTIONS ###########
########################################

LOW = 0
MEDIUM = 1
HIGH = 2

NUM_LIN_UCB_FEATURES = 9

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

# Extract (linUCB) feature vector for a given patient
def get_linUCB_features(patient):
	return get_baseline_linear_features(patient)

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
