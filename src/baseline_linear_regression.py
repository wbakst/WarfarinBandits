import pandas as pd
import numpy as np
from utils import *

data = pd.read_csv('data/warfarin.csv')
data = data[data['Therapeutic Dose of Warfarin'].notnull()]


# In this linear regression baseline we toss out any row with a missing value for height/weight/age
# to be able to follow the model used by the paper

feature_weights = {
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

features = [
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

pred_vals = [0, 0, 0]
true_vals = [0, 0, 0]

num_correct, num_patients = 0, 0
num_discarded = 0
for index, patient in data.iterrows():
  f = get_baseline_linear_features(patient)
  if True in np.isnan(f):
    num_discarded += 1
    continue # Skip rows with missing entries

  num_patients += 1

  pred = 0
  for index, item in enumerate(f):
    pred += item * feature_weights[features[index]]
  pred = pred * pred

  true_dosage = get_true_dosage(patient)

  true_vals[get_dosage_bucket(true_dosage)] += 1

  if correct_predicted_dosage(true_dosage, pred):
    pred_vals[get_dosage_bucket(pred)] += 1
    num_correct += 1

print('total patients', len(data))
print('discarded patients', num_discarded)
print('considered patients', num_patients)
print('accuracy', num_correct, num_patients, num_correct / float(num_patients))
print(pred_vals)
print(true_vals)

