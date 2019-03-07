import pandas as pd
import numpy as np
from utils import *

data = pd.read_csv('data/warfarin.csv')
data = data[data['Therapeutic Dose of Warfarin'].notnull()]


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

num_correct, num_patients = 0, 0
for index, patient in data.iterrows():
  try:
    f = get_baseline_linear_features(patient)
    num_patients += 1
  except:
    continue # skip rows with missing entries

  pred = 0
  for index, item in enumerate(f):
    pred += item * feature_weights[features[index]]
  pred = pred * pred

  if correct_predicted_dosage(get_true_dosage(patient), pred):
    num_correct += 1

print(len(data))
print(num_patients)
print(num_correct / float(num_patients))

