import pandas as pd

data = pd.read_csv("pred.txt", sep=',')
data['tag'] = data['tag'].str.replace('0', 'positive')
data['tag'] = data['tag'].str.replace('1', 'neutral')
data['tag'] = data['tag'].str.replace('2', 'negative')
data.to_csv('test_pred', index=False, sep=',')