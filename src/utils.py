########################################
########## UTILITY FUNCTIONS ###########
########################################

LOW = 0
MEDIUM = 1
HIGH = 2

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