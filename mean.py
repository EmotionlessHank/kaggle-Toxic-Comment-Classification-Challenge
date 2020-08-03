import pandas as pd

data1 = pd.read_csv('submission_MNB_group1.csv')
data2 = pd.read_csv('submission_MNB_group2.csv')
data3 = pd.read_csv('submission_MNB_group3.csv')
data4 = pd.read_csv('submission_MNB_group4.csv')
data5 = pd.read_csv('submission_MNB_group5.csv')

data1['toxic'] = (data1['toxic']+data2['toxic']+data3['toxic']+data4['toxic']+data5['toxic'])/5
data1['severe_toxic'] = (data1['severe_toxic']+data2['severe_toxic']+data3['severe_toxic']+data4['severe_toxic']+data5['severe_toxic'])/5
data1['obscene'] = (data1['obscene']+data2['obscene']+data3['obscene']+data4['obscene']+data5['obscene'])/5
data1['threat'] = (data1['threat']+data2['threat']+data3['threat']+data4['threat']+data5['threat'])/5
data1['insult'] = (data1['insult']+data2['insult']+data3['insult']+data4['insult']+data5['insult'])/5
data1['identity_hate'] = (data1['identity_hate']+data2['identity_hate']+data3['identity_hate']+data4['identity_hate']+data5['identity_hate'])/5
data1.to_csv('submission_MNB_group1to5_mean.csv', index=False)