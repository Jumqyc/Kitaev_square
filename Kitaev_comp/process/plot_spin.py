import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import os

def plot_spin(L,temp,flush):
    path = f'/public/home/hkust_jwliu_1/yqianao/Kitaev_comp/dataAll/N{L}/row1/T{temp}/U_s_dataFiles/s/flushEnd{flush}.s_final.pkl'
    with open(path,'rb') as f:
        s = pkl.load(f)
    s = s.reshape((L,L,3))
    plt.quiver(s[:,:,2],s[:,:,0],s[:,:,1])
    plt.savefig(f'spin_L{L}_T{temp}_flush{flush}.pdf')

plot_spin(32,0.6,100)
