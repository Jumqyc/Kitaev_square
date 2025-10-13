import csv
import numpy as np
import pickle as pkl
import gc
import os
import sys
import fss

from typing import NamedTuple, Callable

result = NamedTuple('result', [('avg', float), ('err', float)])

def evaluate(f:Callable, L:int, t:float, e:np.ndarray, m:np.ndarray) -> result:
    """
    Evaluate the quantity f using the jackknife method.
    Args:
        f: function to evaluate, should take m, e, L, T as arguments.
        L: system size
        t: number of measurements
        e: energy measurements, shape (num_samples, t)
        m: order parameter measurements, shape (num_samples, t)
    Returns:
        result: NamedTuple containing avg and error
    Raises:
        RuntimeError: if jackknife fails
    """
    Q_jack = []
    for i in range(np.size(m,axis=0)):
        index = np.delete(np.arange(np.size(m,axis=0)), i)
        try:
            Q_jack.append(f(m=m[index,:], e=e[index,:], L=L, T=t))
        except RuntimeError:
            raise RuntimeError('Jackknife failed, check the function you provided.')
        finally:
            gc.collect()
            continue
    Q_jack = np.array(Q_jack)
    Q_avg = np.mean(Q_jack)
    Q_err = np.sqrt((np.size(m,axis=1)-1) * np.var(Q_jack, ddof=1))
    gc.collect()
    return result(Q_avg, Q_err)

data_path = '/public/home/hkust_jwliu_1/yqianao/Kitaev_comp/dataAll'


def process_size(size: str):
    print(f'Processing size: {size}')
    if not size.startswith('N'):
        print(f'Invalid size format: {size}. Should start with "N".')
        return
    
    L = int(size[1:])
    csv_path = f'./processed_data/processed_data_{L}.csv'
    os.makedirs('./processed_data', exist_ok=True)

    if os.path.exists(csv_path):
        os.remove(csv_path)

    with open(csv_path, 'x', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['L', 'T',
                        'avg_e', 'err_e', 
                        'specific_heat', 'err_specific_heat',
                        'avg_m', 'err_m', 
                        'susceptibility', 'err_susceptibility',
                        'binder_cumulant', 'err_binder_cumulant',
                        'logd1', 'err_logd1',
                        'logd2', 'err_logd2',
                        'logd3', 'err_logd3',
                        'binderd1', 'err_binderd1',
                        'binderd2', 'err_binderd2',
                        'binderd3', 'err_binderd3'
                     ])
        
    for temperatures in os.listdir(f'{data_path}/{size}/row1'):
        temp = float(temperatures[1:])
        print(size,temp)
        path = f'{data_path}/{size}/row1/{temperatures}/U_s_dataFiles'

        # load order parameters

        order_params_list = []
        energy_list = []
        # load energy
        for flushes in os.listdir(path+'/out_order'):
            try:
                with open(f'{path}/out_order/{flushes}', 'rb') as f:
                    m = pkl.load(f)
                    order_params_list.extend(m)
            except FileExistsError:
                print(f'File {path}/out_order/{flushes} not found, skipping.')
                continue

        for flushes in os.listdir(path+'/U'):
            try:
                with open(f'{path}/U/{flushes}', 'rb') as f:
                    e = pkl.load(f)
                    energy_list.extend(e)
            except FileExistsError:
                continue
        order_params = np.array(order_params_list)
        energy = np.array(energy_list)
        order_params = order_params.reshape((-1,3))
        order_params = order_params[:,2]

        order_params = np.abs(order_params).flatten()
        energy = energy.flatten()
        auto_corr_time = max(fss.rtime(energy),fss.rtime(order_params))
        length = len(energy) // auto_corr_time
        
        energy = energy[:length*auto_corr_time]
        order_params = order_params[:length*auto_corr_time]
        energy = energy.reshape((length,auto_corr_time))
        order_params = order_params.reshape((length,auto_corr_time)) * L**2

        # compute quantities
        energy_result = evaluate(fss.avg_e, L, temp, energy, order_params)
        specific_heat_result = evaluate(fss.specific_heat, L, temp, energy, order_params)
        avg_m_result = evaluate(fss.avg_m, L, temp, energy, order_params)
        susceptibility_result = evaluate(fss.susceptibility, L, temp, energy, order_params)
        binder_cumulant_result = evaluate(fss.binder_cumulant(1), L, temp, energy, order_params)

        logd1_result = evaluate(fss.logd(1), L, temp, energy, order_params)
        logd2_result = evaluate(fss.logd(2), L, temp, energy, order_params)
        logd3_result = evaluate(fss.logd(3), L, temp, energy, order_params)
        binderd1_result = evaluate(fss.binderd(1), L, temp, energy, order_params)
        binderd2_result = evaluate(fss.binderd(2), L, temp, energy, order_params)
        binderd3_result = evaluate(fss.binderd(3), L, temp, energy, order_params)
        # store results
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([L, temp, 
                             energy_result.avg, energy_result.err,
                             specific_heat_result.avg, specific_heat_result.err,
                             avg_m_result.avg, avg_m_result.err,
                             susceptibility_result.avg, susceptibility_result.err,
                             binder_cumulant_result.avg, binder_cumulant_result.err,
                             logd1_result.avg, logd1_result.err,
                             logd2_result.avg, logd2_result.err,
                             logd3_result.avg, logd3_result.err,
                             binderd1_result.avg, binderd1_result.err,
                             binderd2_result.avg, binderd2_result.err,
                             binderd3_result.avg, binderd3_result.err
                             ])
        
        gc.collect()

if __name__ == '__main__':
    process_size(sys.argv[1])