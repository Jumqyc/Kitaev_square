import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def exponent_Tc_fix(data_list: list[tuple[float, float, float]] | np.ndarray, T_c: float, w: float | str) -> tuple[float, float, float]:
    """
    Fit the critical exponents data and plot the results. Universal scaling function F(T) is given in the form of L^{β/ν}M(t) = F(t L^{1/ν}), where t = (T - T_c)/T_c. 

    This function will try to minimize the squared differences between the left-hand side and the right-hand side of the equation by adjusting the parameters β and ν. 
    Parameters:

    T_c: Critical temperature.
    w : Width of the Gaussian function used for weighting the data points. If 'inf', no weighting is applied.

    Returns: 
    A tuple containing the fitted critical exponents:
    (beta,nu,R^2)
    """
    w = float(w)  # Ensure w is a float for calculations

    def gaussian(x,sigma = w):
        return np.exp(-x**2/(2*sigma**2))

    for d in data_list:
        if len(d) != 3:
            raise ValueError("Each data point must be a tuple of (Size, temperature, order parameter).")

    data_list = np.array(data_list,dtype=[('l',int), ('t', float), ('m', float)])
    data_list = np.sort(data_list,order=['l', 't']) 
    data_list['t'] = (data_list['t'] - T_c)/T_c  # Normalize temperature by T_c

    size_list = data_list['l']
    # select most frequent size
    feq_size = data_list[np.bincount(size_list.astype(int)).argmax()]

    base_data = data_list[data_list['l'] == feq_size['l']]  # select the data with the most frequent size
    # we will use this as the source for universal scaling function F(t L^{1/ν})

    def Rsqr(b,nu, w) -> float:
        scaled_data_list = data_list.copy()
        scaled_data_list['t'] = data_list['t'] * (data_list['l']**(1/nu))
        scaled_data_list['m'] = data_list['m'] * (data_list['l']**(b/nu))
        
        f = interp1d(
            base_data['t'] * (base_data['l']**(1/nu)), 
            base_data['m'] * (base_data['l']**(b/nu)), 
            bounds_error=False) # interpolate the universal scaling function F(t L^{1/ν})
        if w < float('inf'):
            return np.nansum(
                ((scaled_data_list['m'] - f(scaled_data_list['t']))**2) 
                * gaussian(scaled_data_list['t'])
                )
        else:
            return np.nansum(
                ((scaled_data_list['m'] - f(scaled_data_list['t']))**2)
            )


    min_R2 = float('inf')
    best_b = 0.0
    best_nu = 0.0

    for b_val in np.linspace(0.11, 3, 30):
        for nu_val in np.linspace(0.11, 3, 30):
            current_R2 = Rsqr(b_val, nu_val,w = float('inf'))
            
            if current_R2 < min_R2:
                min_R2 = current_R2
                best_b = b_val
                best_nu = nu_val
                print(f"New best: b={best_b}, nu={best_nu}, R^2={min_R2}", end='\r')
    
    if w == float('inf'):
        return best_b, best_nu, min_R2

    for b_val in np.linspace(best_b-0.1, best_b+0.1, 10):
        for nu_val in np.linspace(best_nu-0.1, best_nu+0.1, 10):
            current_R2 = Rsqr(b_val, nu_val, w = w)
            if current_R2 < min_R2:
                min_R2 = current_R2
                best_b = b_val
                best_nu = nu_val
                print(f"New best: b={best_b}, nu={best_nu}, R^2={min_R2}", end='\r')

    plt.plot(data_list['t'] * (data_list['l']**(1/best_nu)), 
             data_list['m'] * (data_list['l']**(best_b/best_nu)), 'o', label='Data')
    x_val = np.linspace(-5*w, 5*w, 1000)

    plt.plot(x_val,gaussian(x_val, w))
    return best_b, best_nu, min_R2

