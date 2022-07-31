"""
SOFTA == software for first time arrivals
this script is reading pickings txt-files exported from RadExPro
and is aimed to observe them and convert them to different formats

also this script is aimed at crosswell seismic tomography data specifically

! here's a convention: RadExPro picking file is ending with '__f.txt', where
f is a distance between wells, and it is the only place, there '__' takes
place in the filename

todo:
    check if convention is being respected (and show angry message if not)
    add an option to just load all .txt files in current folder w/o openFileDialog
    actually get headers from files and use them to name pd columns
    
"""
# import numpy as np
import pandas as pd
# import math
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo

# %% parsing lines in pickings txt-files exported from RadExPro
def parse_line(line):
    tokens = line.strip().split(' ')
    depth, rec_elev, time = [float(t.replace(':', '')) for t in tokens if t != '']
    return depth, rec_elev, time
# %%
# FILE_NAME = 's6_r2_FBA_pick.txt'
root = tk.Tk()
root.withdraw()

filetypes = ( ('text files', '*.txt'), ('All files', '*.*') )
file_paths = fd.askopenfilenames( 
					title='Thy bidding, master?', initialdir=os.getcwd(), 
                    filetypes=filetypes)

# showinfo(title='I bow to your will.', message=file_paths)
# %% manual reading...
depths, rec_elevs, times = [], [], []
for file_path in file_paths:
    with open(file_path) as file_handler:
        file_handler.readline() #skip first line with headers
        for line in file_handler:
            d,r,t = parse_line(line)
            depths.append(d)
            rec_elevs.append(r)
            times.append(t)
        pass #here comes some action to save lists
# %%
# fig, ax = plt.subplots(nrows=1, ncols=1)#, figsize=(5, 3))
# ax.plot(
#     rec_elevs,times,
#     color='blue', linestyle='', marker='.', mew=0.5
# )
# ax.set_xlabel('rec_elev, m')
# ax.set_ylabel('traveltime, ms')
# %%
# DISTANCE = 88.5
# picks_li = []
picks_di = {}
for file_path in file_paths:
    filename = file_path.split('/')[-1]
    pickname, distance_txt = filename.split('__')
    distance = float(distance_txt[:-4])
    df = pd.read_csv(file_path)
    
    # picks_li.append(df)
    picks_di[pickname] = df

# %%
# for x in dict: #itarate over keys; values are just dict[x]; or itarate over dict.values()
for pickname in picks_di:
    df = picks_di[pickname] 
    Vel = 2000 #average model velocity (crutch: should be calculated)
    # new data frame with split value columns:
    new = df["SOU_ELEV:REC_ELEV"].str.split(":", n = 1, expand = True)     
    
    df['Depth'] = pd.to_numeric(new[0], errors='coerce')
    new2 = new[1].str.split(expand = True)
    df['Rec_Elev'] = pd.to_numeric(new2[0])
    df['time'] = pd.to_numeric(new2[1])
    df = df.drop("SOU_ELEV:REC_ELEV", axis=1)
    
    df['st.ray_len'] =  (distance**2 + (df['Rec_Elev']-df['Depth'])**2)**(0.5)
    
    # df['angle'] = abs(df['Rec_Elev']- df['Depth'])
    
    df['reduced_time'] = df['time'] - (df['st.ray_len']/Vel)
    
    df['calc.ray_len'] = df['time'] * Vel
    
    df['time, s'] = df['time']/1000 #converting time from [ms] to [s]
# %%
    options = [df['Rec_Elev'], df['Depth'], df['st.ray_len'] ]
    fig, axs = plt.subplots(nrows=3, ncols=1, 
                            figsize=(5, 8), constrained_layout = True)
    for pl_Num, ax in enumerate(axs):
        ax.plot(
        options[pl_Num], df['time'],
        color='blue', linestyle='', marker='.', mew=0.5, 
    )
        ax.set_xlabel('depth, m')
        ax.set_ylabel('traveltime, ms')
        
        fig.suptitle(pickname, fontsize=20, color="green")
        
        plt.savefig(pickname + '.png')
# %%
showinfo(title='I gladly obey.', message='It is done.')
# %% Drawing control plots
# checklist (Lehman):
# A1
# A2
# A3 is basically the same as A4; is this always like that? Oh. It's not!
# A4 -/-
# B1
# B2 
# C1
# C2
# D1
# D2
# A: traveltime
# B: reduced_traveltime = measured_traveltime - (distance_along_ray/average_velocity_for_model)
# C: calc. ray length = measured_traveltime * average_velocity_for_model
# D: average velocity
# 1: ray length
# 2: angle
# 3: shot no.
# 4: receiver no.