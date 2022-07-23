from numpy import ScalarType
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer

df = pd.read_excel('dataframe.xlsx')

scaler = StandardScaler()
sta_df = scaler.fit_transform(df)

scaler = MinMaxScaler()
mm_df = scaler.fit_transform(df)

scaler = RobustScaler()
rob_df = scaler.fit_transform(df)

scaler = Normalizer()
nom_df = scaler.fit_transform(df)

sta_df = pd.DataFrame(sta_df)
mm_df = pd.DataFrame(mm_df)
rob_df = pd.DataFrame(rob_df)
nom_df = pd.DataFrame(nom_df)

data_list = []
sta_data_list =[]
mm_data_list =[]
rob_data_list =[]
nom_data_list =[]

for i in range(len(df.columns)-1):
    data_list.append(df.iloc[:,i:i+1])

for i in range(len(df.columns)-1):
    sta_data_list.append(sta_df.iloc[:,i:i+1])

for i in range(len(df.columns)-1):
    mm_data_list.append(mm_df.iloc[:,i:i+1])

for i in range(len(df.columns)-1):
    rob_data_list.append(rob_df.iloc[:,i:i+1])

for i in range(len(df.columns)-1):
    nom_data_list.append(nom_df.iloc[:,i:i+1])

f, axes = plt.subplots(6, 5)
print(axes)
f.set_size_inches((10, 7))
plt.subplots_adjust(wspace = 0.3, hspace = 0.3)

for i in range(len(df.columns) - 1):
    axes[1, i].plot(range(len(df)), data_list[i])

for i in range(len(df.columns) - 1):
    axes[2, i].plot(range(len(df)), sta_data_list[i])

for i in range(len(df.columns) - 1):
    axes[3, i].plot(range(len(df)), mm_data_list[i])

for i in range(len(df.columns) - 1):
    axes[4, i].plot(range(len(df)), rob_data_list[i])

for i in range(len(df.columns) - 1):
    axes[5, i].plot(range(len(df)), nom_data_list[i])
    
plt.show()
