import pandas as pd
import numpy as np

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

def get_amiodorone_status(row):
  medications = row['Medications']
  if medications is 'nan':
    print 'got here'
  if 'amiodarone' in medications and 'not amiodarone' not in medications:
    return 1
  return 0

def get_enzyme_status(row):
  num_inducers = row['Rifampin or Rifampicin'] + row['Carbamazepine (Tegretol)'] + row['Phenytoin (Dilantin)']
  return 1 if num_inducers > 0 else 0



# impute data values

for index, row in data.iterrows():
  age = int(row['Age'][0])
  height = row['Height (cm)']
  weight = row['Weight (kg)']
  race = row['Race']

  asian = 1 if race is 'Asian' else 0
  black = 1 if race is 'Black' else 0
  missing = 1 if race is 'Unknown' else 0

  enzyme = get_enzyme_status(row)
  amiodorone_status = get_amiodorone_status(row)
  f = [1, age, height, weight, asian, black, missing, enzyme, amiodorone_status]
  pred = 0
  for index, item in enumerate(f):
    pred += item * feature_weights[features[index]]
  pred = pred * pred
  print pred