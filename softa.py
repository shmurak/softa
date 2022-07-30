"""
SOFTA == software for first time arrivals
this script is reading pickings txt-files exported from RadExPro
and is aimed to observe them and convert them to different formats
"""
# import numpy as np
import pandas as pd
# import math
import matplotlib.pyplot as plt
# %%
def parse_line(line):
    tokens = line.strip().split(' ')
    depth, rec_elev, time = [float(t.replace(':', '')) for t in tokens if t != '']
    return depth, rec_elev, time
# %%
FILE_NAME = 's6_r2_FBA_pick.txt'
depths, rec_elevs, times = [], [], []
with open(FILE_NAME) as file_handler:
    file_handler.readline()#пропускаем первую строчку с названиями
    for line in file_handler:
        threesome = parse_line(line)
        depths.append(threesome[0])
        rec_elevs.append(threesome[1])
        times.append(threesome[2])
# %%
# fig, ax = plt.subplots(nrows=1, ncols=1)#, figsize=(5, 3))
# ax.plot(
#     rec_elevs,times,
#     color='blue', linestyle='', marker='.', mew=0.5
# )
# ax.set_xlabel('rec_elev, m')
# ax.set_ylabel('traveltime, ms')
# %%
options = [rec_elevs, depths]
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(5, 3))
for pltNum, ax in enumerate(axs):
    ax.plot(
    options[pltNum], times,
    color='blue', linestyle='', marker='.', mew=0.5
)
    ax.set_xlabel('depth, m')
    ax.set_ylabel('traveltime, ms')
# plt.show()
# %%
DISTANCE = 88.5
df = pd.read_csv(FILE_NAME)
# new data frame with split value columns:
new = df["SOU_ELEV:REC_ELEV"].str.split(":", n = 1, expand = True) 
# %%
# new.head()
# # it works:
depth = pd.to_numeric(new[0], errors='coerce')
df['Depth'] = depth
# new2 = new[1].str.split(" ", n = 20, expand = True)
# new2 = new[1].str.split(n = 20, expand = True)
new2 = new[1].str.split(expand = True)
df['Rec_Elev'] = pd.to_numeric(new2[0])
df['time'] = pd.to_numeric(new2[1])
df = df.drop("SOU_ELEV:REC_ELEV", axis=1)
# df['x2'] = DISTANCE**2
# df['z2'] = (df['Rec_Elev']- df['Depth'])**2
# df['dist'] =  (df['x2'])**(0.5)
df['dist'] =  (DISTANCE**2 + (df['Rec_Elev']-df['Depth'])**2)**(0.5)
df['time, s'] = df['time']/1000 #converting time from [ms] to [s]
# df.head()