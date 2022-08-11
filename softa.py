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
    
    paint data points on graphs according to their shot numbers (and add colorbar)
    
"""
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression #scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tkinter as tk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo


# parsing lines in pickings txt-files exported from RadExPro
def parse_line(line):
    tokens = line.strip().split(' ')
    depth, rec_elev, time = [float(t.replace(':', '')) for t in tokens if t != '']
    return depth, rec_elev, time

def linear_regression(X, y):
    SEED = 42
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                        random_state=SEED)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    b, k = regressor.intercept_, regressor.coef_
    k = k[0][0]
    b = b[0] #this should be zero (but it's not)

    y_pred = regressor.predict(X_test)
    df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred.squeeze()})
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f'k:{k:.2f}, b:{b:.2f}')
    print(f'Mean absolute error: {mae:.2f}')
    print(f'Mean squared error: {mse:.2f}')
    print(f'Root mean squared error: {rmse:.2f}')

    return k, b
    # P1_x, P1_y = df['Depth'], 0
    # P2_x, P2_y = df['Rec_Elev'], distance
    # deltaY = P2_y - P1_y
    # deltaX = P2_x - P1_x
    # (deltaX, deltaY) is a vector of ray
    # unit_vec = np.linalg.norm([deltaX,deltaY])
    # unit_vec = np.linalg.norm([df['Rec_Elev'] - df['Depth'], distance])


# FILE_NAME = 's6_r2_FBA_pick.txt'
root = tk.Tk()
root.withdraw()

filetypes = (('text files', '*.txt'), ('All files', '*.*'))
file_paths = fd.askopenfilenames(
    title='Thy bidding, master?', initialdir=os.getcwd(),
    filetypes=filetypes)


########################################################################################

########################################################################################

########################################################################################

# OpenFileDialog for RadExPicks and getting all data into pandas DataFrames

root = tk.Tk() # I genuinly don't know that these 2 lines do
root.withdraw()# But one day I'll know, I promise

filetypes = (('text files', '*.txt'), ('All files', '*.*'))
file_paths = fd.askopenfilenames(
    title='Thy bidding, master?', initialdir=os.getcwd(),
    filetypes=filetypes)

picks_di = {}

for file_path in file_paths:
    filename = file_path.split('/')[-1]
    pickname, distance_txt = filename.split('__')
    df = pd.read_csv(file_path)
    df['Distance'] = float(distance_txt[:-4])
    picks_di[pickname] = df


########################################################################################

########################################################################################

########################################################################################

# Calculating necessary attributes for all pandas DataFrames in picks_di

# for x in dict: #itarate over keys; values are just dict[x]; or itarate over dict.values()
for pickname in picks_di: 
    df = picks_di[pickname]
    first_line = df.columns[0] #"SOU_ELEV:REC_ELEV" by default, but sometimes 'SOU_X:REC_ELEV','SOU_Y:REC_ELEV'
    # new data frame with split value columns:
    tmp = df[first_line].str.split(":", n=1, expand=True)

    df['Depth'] = pd.to_numeric(tmp[0], errors='coerce')

    tmp2 = tmp[1].str.split(expand=True)

    df['Rec_Elev'] = pd.to_numeric(tmp2[0])
    df['time'] = pd.to_numeric(tmp2[1])
    df = df.drop(first_line, axis=1)


    df['st.ray_len'] = (df['Distance'] ** 2 + (df['Rec_Elev'] - df['Depth']) ** 2) ** 0.5

    #linear regression    
    k, b = linear_regression(X = df['st.ray_len'].values.reshape(-1, 1), y = df['time'].values.reshape(-1, 1))
    Vel = 1/k #* 1000  # average model velocity (km/s)

    df['angle'] = np.arctan2(df['Distance'],
                             df['Rec_Elev'] - df['Depth']
                             ) * 180 / math.pi

    df['average velocity'] = df['st.ray_len'] / df['time']
    df['reduced_time'] = df['time'] - (df['st.ray_len'] / Vel)

    df['calc.ray_len'] = df['time'] * Vel

    df['time, s'] = df['time'] / 1000  # converting time from [ms] to [s]
    
    picks_di[pickname] = df 


########################################################################################

########################################################################################

########################################################################################

# Draw Graphs for all picks (which are represented as separate values in picks_di)
x_axs = ['st.ray_len', 'angle', 'Depth', 'Rec_Elev']
y_axs = ['time', 'reduced_time', 'calc.ray_len', 'average velocity']

indexes = [(i, j) for i in range(2) for j in range(4)]
indexes.append((2, 0)), indexes.append((3, 0))

for pickname in picks_di:
    df = picks_di[pickname]

    fig, axs = plt.subplots(nrows=4, ncols=4,
                            figsize=(8, 8), constrained_layout=True)

    for ind in indexes:
        i, j = ind
        axs[i, j].plot(
            df[x_axs[i]], df[y_axs[j]],  # options[pl_Num], df['time'],
            color='blue', linestyle='', marker='.', mew=0.5,
        )
        axs[i, j].set_xlabel(x_axs[i])
        axs[i, j].set_ylabel(y_axs[j])
        if (i, j) == (0, 0):
            x_min = min(df[x_axs[i]])
            x_max = max(df[x_axs[i]])
            print(f'pick={pickname}, k={k}, b={b}') #debugging
            axs[i, j].plot(
                [x_min, x_max],
                [x_min * k + b, x_max * k + b],
                color='red'
            )

    fig.suptitle(pickname, fontsize=20, color="green")

    plt.savefig(pickname + '.png')
print('DONE')
# showinfo(title='I gladly obey.', message='It is done.')


########################################################################################

########################################################################################

########################################################################################

# exporting pickings as *.3dd files with columns num	sou_x	sou_y	sou_z	rec_x	rec_y	rec_z	traveltime
# this is 2d version
for pickname in picks_di:
    df = picks_di[pickname]
    df.index.name = 'num' 
    df.index += 1 #index should be starting with 1
 

    df['sou_x'] = 0
    df['sou_y'] = df['Depth']
    df['sou_z'] = 0
    
    df['rec_x'] = df['Distance']
    df['rec_y'] = df['Rec_Elev']
    df['rec_z'] = 0
    
    df['traveltime'] = df['time']
    culumnsFor3dd = df.columns.values.tolist()[-7:] #last 7 columns that just've been added
    df = df[culumnsFor3dd] #df = df[['a','b']] - here comes list of desired columns
    
    # clunky way of adding empty line
    df_empty = pd.DataFrame([[np.nan] * len(df.columns)], columns=df.columns)
    df = pd.concat([df_empty, df])
#     df = df_empty.append(df, ignore_index=True)
    df.to_csv(pickname + ".3dd", sep = '	', index_label='num')


# # checklist (Lehman):
# # A1
# # A2
# # A3 is basically the same as A4; is this always like that? Oh. It's not!
# # A4 -/-
# # B1
# # B2 
# # C1
# # C2
# # D1
# # D2
# # A: traveltime
# # B: reduced_traveltime = measured_traveltime - (distance_along_ray/average_velocity_for_model)
# # C: calc. ray length = measured_traveltime * average_velocity_for_model
# # D: average velocity
# # 1: ray length
# # 2: angle
# # 3: shot no.
# # 4: receiver no.
