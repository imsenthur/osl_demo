#Load plotting library
from rtplot import client 

#Common import
import numpy as np

#Let's create two subplots
#First, define a dictionary of items for each plot

#First plot will have three traces: phase, phase_dot, stride_length
plot_1_config = {'names': ['phase', 'phase_dot', 'stride_length'],
                 'title': "Phase, Phase Dot, Stride Length",
                 'ylabel': "reading (unitless)",
                 'xlabel': 'test 1'}

#Second plot will have five traces: gf1, gf2, gf3, gf4, gf5
plot_2_config = {'names': ["gf1","gf2","gf3","gf4","gf5"],
                 'colors' : ["r","b","g","r","b"],
                 'line_style' : ['-','','-','','-'],
                 'title': "Phase, Phase Dot, Stride Length",
                 'ylabel': "reading (unitless)",
                 'xlabel': 'test 2'}

#Aggregate into list  
plot_config = [plot_1_config,plot_2_config]

#Tell the server to initialize the plot
client.initialize_plots(plot_config)

#Create plotter array with random data
plot_data_array = [np.random.randn(), #phase
                   np.random.randn(), #phase_dot
                   np.random.randn(), #stride_length
                   np.random.randn(), #gf1
                   np.random.randn(), #gf2
                   np.random.randn(), #gf3
                   np.random.randn(), #gf4
                   np.random.randn()  #gf5
                   ]

#Send data to server to plot
client.send_array(plot_data_array)