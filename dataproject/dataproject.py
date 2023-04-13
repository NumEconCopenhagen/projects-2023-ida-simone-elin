import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import plotly.graph_objects as go
import ipywidgets as widgets


#Cleaning the 4 datasets simultaneously
def dataclean(df,value):
   """function that drops and renames columns"""
   drop_these = ['INDICATOR','MEASURE','FREQUENCY','Flag Codes'] 
   df.drop(drop_these, axis=1, inplace=True)
   df.rename(columns={'LOCATION':'country', 'TIME':'year', 'Value':value}, inplace=True)   
   return df





