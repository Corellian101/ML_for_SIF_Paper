import numpy as np
import pandas as pd

# Loading dataset
df1 = pd.read_csv("../../data/3_datasets_new/train/1_RN_data_TRAIN.csv")
df1.drop(["c/b", "F"], axis=1, inplace=True)
d1 = df1.to_numpy()

df2 = pd.read_csv("../../data/3_datasets_new/train/2_RN_eqn_TRAIN.csv")
df2.drop(["c/b", "F"], axis=1, inplace=True)
d2 = df2.to_numpy()

df3a = pd.read_csv("../../data/3_datasets_new/train/3_FRANC3D_FULL_TRAIN.csv")
df3a.drop(["c/b", "F"], axis=1, inplace=True)
d3a = df3a.to_numpy()

df3b = pd.read_csv("../../data/3_datasets_new/train/3_FRANC3D_PHI_SAMPLE_TRAIN.csv")
df3b.drop(["c/b", "F"], axis=1, inplace=True)
d3b = df3b.to_numpy()

# Splitting a/c
d1_le1 = []
d1_g1 = []
for ex in d1:
    if ex[0] <= 1:
        d1_le1.append(ex)
    else:
        d1_g1.append(ex)

np.save("1_RN_data_ac_le1.npy", d1_le1)
np.save("1_RN_data_ac_g1.npy", d1_g1)


d2_le1 = []
d2_g1 = []
for ex in d2:
    if ex[0] <= 1:
        d2_le1.append(ex)
    else:
        d2_g1.append(ex)

np.save("2_RN_eqn_ac_le1.npy", d2_le1)
np.save("2_RN_eqn_ac_g1.npy", d2_g1)

d3a_le1 = []
d3a_g1 = []
for ex in d3a:
    if ex[0] <= 1:
        d3a_le1.append(ex)
    else:
        d3a_g1.append(ex)

np.save("3a_FULL_data_ac_le1.npy", d3a_le1)
np.save("3a_FULL_data_ac_g1.npy", d3a_g1)

d3b_le1 = []
d3b_g1 = []
for ex in d3b:
    if ex[0] <= 1:
        d3b_le1.append(ex)
    else:
        d3b_g1.append(ex)

np.save("3b_PHI_data_ac_le1.npy", d3b_le1)
np.save("3b_PHI_data_ac_g1.npy", d3b_g1)
