# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 20:45:52 2017

@author: SamKao
"""

import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv('cDNI_numbers.csv', header = None).values
df2 = pd.read_csv('cDNI_edit_numbers.csv', header = None).values
df3 = pd.read_csv('feed_forward_NN_numbers.csv', header = None).values
plt.plot(df1, 'r', label = 'cDNI')
plt.plot(df2, 'g', label = 'cDNI_edit')
plt.plot(df3, 'b', label = 'feed_forward_NN')
plt.legend(loc= 'upper right')