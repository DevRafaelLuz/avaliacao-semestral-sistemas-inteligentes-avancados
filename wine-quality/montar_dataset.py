import pandas as pd

df1 = pd.read_csv('wine-quality/winequality-red.csv')
df2 = pd.read_csv('wine-quality/winequality-white.csv')

df_final = pd.concat([df1, df2], axis=0, join='outer', ignore_index=True)

df_final.to_csv('wine-quality/winequality.csv', index=False)