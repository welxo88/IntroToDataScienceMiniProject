import librosa
import pandas as pd
import os
import time

start = time.clock()
files = os.listdir(
    ".\data\cleaned")
data = pd.Series([librosa.util.pad_center(librosa.load(
    ".\data\cleaned\\"+x, mono=True)[0], 88375) for x in files])
df = pd.DataFrame()

df['data'] = data
df['label'] = 'gun_shot'

print('Doing work, hold tight')
labels = pd.read_csv(".\\train\\train.csv")
files2 = os.listdir(".\\train\\train")
data2 = pd.Series([librosa.util.pad_center(librosa.load(
    ".\\train\\train\\"+x, mono=True)[0], 88375) for x in files2])

df2 = pd.DataFrame()
df2['data'] = data2
df2['label'] = labels['Class']

df = df.append(df2)
df.to_pickle('dataset.pkl')
print('Done')

end = time.clock()
print("Took %.1f minutes" % ((end-start)/60))
