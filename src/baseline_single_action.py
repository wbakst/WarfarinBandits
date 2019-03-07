import pandas as pd
import numpy as np

data = pd.read_csv('data/warfarin.csv')
data = data[data['Therapeutic Dose of Warfarin'].notnull()]

# Basline 1
warf = data.loc[:,'Therapeutic Dose of Warfarin'].values.tolist()
baseline1 = sum([1. for i in warf if i >= 21 and i <= 49]) / len(warf) # 3-5 mg/day
print(baseline1)