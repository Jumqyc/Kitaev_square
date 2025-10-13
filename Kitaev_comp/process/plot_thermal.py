
import matplotlib.pyplot as plt
import pandas as pd
import os

plot_path = '/public/home/hkust_jwliu_1/yqianao/Kitaev_comp/plot'
data_list = []
for file in os.listdir('./'):
    if file.startswith('processed_data_') and file.endswith('.csv'):
        data = pd.read_csv(f'./{file}')
        data_list.append(data)


data = pd.concat(data_list, ignore_index=True)
data.columns = ['L', 'T', 'avg_e', 'err_e', 'specific_heat', 'err_specific_heat',
                     'avg_m', 'err_m', 'susceptibility', 'err_susceptibility',
                     'binder_cumulant', 'err_binder_cumulant']

# plot energy with error bars
for L in data['L'].unique():
    subset = data[data['L'] == L]
    plt.errorbar(subset['T'], subset['avg_e'], yerr=subset['err_e'], label=f'L={L}', capsize=3, marker='o',ls = '',lw = 2)
plt.xlabel('Temperature T')
plt.ylabel('Average Energy per Site ⟨e⟩')
plt.title('Average Energy per Site vs Temperature')
plt.legend()
plt.grid()
plt.savefig(f'{plot_path}/avg_energy.pdf')

plt.clf()
# plot specific heat with error bars
for L in data['L'].unique():
    subset = data[data['L'] == L]
    plt.errorbar(subset['T'], subset['specific_heat'], yerr=subset['err_specific_heat'], label=f'L={L}', capsize=3, marker='o',ls = '',lw = 2)
plt.xlabel('Temperature T')
plt.xlim(0.85,1.05)
plt.ylabel('Specific Heat C')
plt.title('Specific Heat vs Temperature')
plt.legend()
plt.grid()
plt.savefig(f'{plot_path}/specific_heat.pdf')
plt.clf()
# plot magnetization with error bars
for L in data['L'].unique():
    subset = data[data['L'] == L]
    plt.errorbar(subset['T'], subset['avg_m'], yerr=subset['err_m'], label=f'L={L}', capsize=3, marker='o',ls = '',lw = 2)
plt.xlabel('Temperature T')
plt.ylabel('Average Magnetization per Site ⟨m⟩')
plt.title('Average Magnetization per Site vs Temperature')
plt.legend()
plt.grid()
plt.savefig(f'{plot_path}/avg_magnetization.pdf')
plt.clf()
# plot susceptibility with error bars
for L in data['L'].unique():
    subset = data[data['L'] == L]
    plt.errorbar(subset['T'], subset['susceptibility'], yerr=subset['err_susceptibility'], label=f'L={L}', capsize=3, marker='o',ls = '',lw = 2)
plt.xlabel('Temperature T')
plt.ylabel('Magnetic Susceptibility χ')
plt.title('Magnetic Susceptibility vs Temperature')
plt.legend()
plt.grid()
plt.savefig(f'{plot_path}/susceptibility.pdf')
plt.clf()
# plot binder cumulant with error bars
for L in data['L'].unique():
    subset = data[data['L'] == L]
    plt.errorbar(subset['T'], subset['binder_cumulant'], yerr=subset['err_binder_cumulant'], label=f'L={L}', capsize=3, marker='o',ls = '',lw = 2)
plt.xlabel('Temperature T')
plt.ylabel('Binder Cumulant U')
plt.title('Binder Cumulant vs Temperature')
plt.legend()
plt.grid()
plt.savefig(f'{plot_path}/binder_cumulant.pdf')
plt.clf()

