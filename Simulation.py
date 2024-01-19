#Libraries needed to run the tool
import numpy as np
import pandas as pd
from numpy import mean
from sklearn.neural_network import MLPRegressor
import pickle
import matplotlib.pyplot as plt

# read file with original dataset to get values for minmax scaling
harv = pd.read_excel('Harvesting data.xlsx')
water = harv['Water_volume']
harv = harv.drop(columns = ['Water_volume'], axis = 1)
minval = harv.min()
minmax = harv.max() - harv.min()
minval_water = water.min()
minmax_water = water.max() - water.min()

# read file with operating conditions for the simulation
data = pd.read_excel('Simulation data.xlsx', sheet_name='Sheet1')
time = data['Time']
temp = data['Temperature']
rh = data['RH']
data = data.drop(columns = ['Time'], axis = 1)

data_scaled = (data - minval) / minmax

# ask for model to use
name = input('Model to use for prediction (gbr, mlp, svr): \n')

if name == 'gbr':
    model = pickle.load(open('gbm.pkl', 'rb'))
    
if name == 'mlp':
    model = pickle.load(open('nn.pkl', 'rb'))
    
if name == 'svr':
    model = pickle.load(open('svr.pkl', 'rb'))

results_scaled = nn.predict(data_scaled)

# unscale the predicted values 
results = (results_scaled * minmax_water + minval_water)*60/50

cumulative = np.zeros(len(results))

for i in range(1,len(results)):
    cumulative[i] = cumulative[i-1] + results[i-1]
    
    
# plot
time = [f"{hour:02d}:00" for hour in range(25)]

plt.rcParams.update({'figure.figsize': [8,6],
                     'lines.markersize': 10.0})
fig, ax = plt.subplots()
twin1 = ax.twinx()
twin2 = ax.twinx()
twin2.spines.right.set_position(("axes", 1.15))

p1, = ax.plot(time, cumulative, "o-", c = 'tab:blue', label="Collected volume", alpha = 0.7)
p2, = twin1.plot(time, temp, "D-", c = 'tab:orange', label="Temperature", alpha = 0.7)
p3, = twin2.plot(time, rh, "s-", c = 'tab:green', label="RH", alpha = 0.7)

ax.set_ylim(0, 140)
twin1.set_ylim(20, 40)
twin2.set_ylim(0, 60)

ax.set_xlabel("Time", size = 20)
ax.set_xticklabels(time, rotation = 'vertical', size = 15)
ax.set_ylabel("Collected volume [ml]", size = 20)
twin1.set_ylabel("Temperature [°C]", size = 20)
twin2.set_ylabel("Relative humidity [%]", size = 20)
ax.tick_params(axis='both', which='major', labelsize=15)
twin1.tick_params(axis='both', which='major', labelsize=15)
twin2.tick_params(axis='both', which='major', labelsize=15)

ax.yaxis.label.set_color(p1.get_color())
twin1.yaxis.label.set_color(p2.get_color())
twin2.yaxis.label.set_color(p3.get_color())

tkw = dict(size=4, width=1.5)
ax.tick_params(axis='y', colors=p1.get_color(), **tkw)
twin1.tick_params(axis='y', colors=p2.get_color(), **tkw)
twin2.tick_params(axis='y', colors=p3.get_color(), **tkw)
ax.tick_params(axis='x', **tkw)

ax.legend(handles=[p1, p2, p3], loc='lower right',
          fontsize = 15)

plt.savefig('simulation.svg', bbox_inches='tight')
plt.show
