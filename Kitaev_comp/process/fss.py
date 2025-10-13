from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable , NamedTuple
from scipy.stats import linregress
from scipy.signal import correlate

raw_data = NamedTuple('raw_data', [('T', float), ('M', np.ndarray),('E', np.ndarray)])
result = NamedTuple('result', [("T",float),('avg', float), ('err', float)])

    # Built-in functions for common calculations
def entire_logd(k: int,m:np.ndarray, e:np.ndarray, L:int, T:float)-> np.ndarray:
    return (np.mean((m**k)*e)/ np.mean(m**k) - np.mean(e))/L**4
    # <m^k *e >/ <m^k> - <e>
def logd(k:int)-> Callable:
    return partial(entire_logd, k=k)
def entire_binderd(k: int,m:np.ndarray, e:np.ndarray, L:int, T:float)-> np.ndarray:
    return -(1/3)* (
        (np.mean(m**(2*k) * e) - np.mean(e) * np.mean(m**(2*k))) /
            (np.mean(m**k)**2) 
        -2 * np.mean(m**(2*k)) 
            * (np.mean(e * m**k) - np.mean(e) * np.mean(m**k)) /
            np.mean(m**k)**3
        )/T**2
        # ((<e* m^2k> - <e>*<m^2k>) / <m^k>^2 - 2 <e* m^2k>(<e m^k>-<e>*<m^k>) / <m^k>^3)/T**2
def binderd(k:int) -> Callable:
    return partial(entire_binderd, k=k)
def avg_m(m:np.ndarray, e:np.ndarray, L:int, T:float)-> np.ndarray:
    return np.mean(np.abs(m))/L**2
def avg_e(m:np.ndarray, e:np.ndarray, L:int, T:float)-> np.ndarray:
    return np.mean(e)/L**2
def susceptibility(m:np.ndarray, e:np.ndarray, L:int, T:float)-> np.ndarray:
    return (np.mean(m**2) - np.mean(np.abs(m))**2)/L**2/T
def specific_heat(m:np.ndarray, e:np.ndarray, L:int, T:float)-> np.ndarray:
    return (np.mean(e**2) - np.mean(e)**2)/L**2/T**2
def entire_binder_cumulant(k:int,m:np.ndarray, e:np.ndarray, L:int, T:float)-> np.ndarray:
    return 1 - np.mean(m**(4*k)) / (3 * np.mean(m**(2*k))**2)
def binder_cumulant(k:int)-> Callable:
    return partial(entire_binder_cumulant, k=k)

def rtime(data: np.ndarray) -> int:
    data = data.flatten().astype(float)
    N = len(data)
    if N < 10:
        raise ValueError("Time series too short to estimate autocorrelation time.")
    data = data - np.mean(data)

    # compute full autocovariance, take non-negative lags (N entries)
    acov = correlate(data, data, mode='full')[N-1:] / (N - np.arange(N))
    if acov[0] == 0:
        raise ValueError("Zero variance in time series; cannot compute ACF.")
    acf = acov / acov[0]

    neg_idx = np.where(acf < 0)[0]
    if neg_idx.size != 0:
        # first negative lag after k=0
        k_max = int(neg_idx[0])
        tau_int = 0.5 + np.sum(acf[1:k_max+1])
        rtime = int(0.5 + tau_int) + 1
        return 6 * rtime
    else:
        raise ValueError("Failed to find negative autocorrelation. Time series may be too short.")



class fss:
    def __init__(self) -> None:
        """
        A python module to handle finite size scaling data.
        It can read data from pickle files, process it, and plot the results.
        """
        self.raw :dict[int, list[raw_data]] = {}
        self.processed :dict[int,list[result]]= {}
        self.T_c = 0
        self.a_best = 0
        self.b_best = 0
        self.nu = 1
            

    def load_raw_data(self,ext_e_data:np.ndarray,ext_m_data:np.ndarray,T:float,L:int,block_size:int = 20):
        """
        Add raw data to this object. Please add the data one by one, not all at once.
        Args:
            ext_e_data (np.ndarray): The energy data to be added.
            ext_m_data (np.ndarray): The magnetization data to be added.
            T (float): The temperature at which the data was collected.
            L (int): The system size.
        Raises:
            ValueError: If the length of energy and magnetization data is not the same.
        """
        if len(ext_e_data) != len(ext_m_data):
            raise ValueError("The length of energy and magnetization data must be the same.")
        if L not in self.raw.keys():
            self.raw[L] = []
        max_rtime = max(rtime(ext_m_data), rtime(ext_e_data))

        num = len(ext_e_data)

        ext_e_data = ext_e_data[:max_rtime * (num // max_rtime)]
        ext_m_data = ext_m_data[:max_rtime * (num // max_rtime)]

        ext_e_data = ext_e_data.reshape((num//max_rtime,max_rtime)) # reshape to (max_rtime, num/max_rtime). 
        ext_m_data = ext_m_data.reshape((num//max_rtime,max_rtime))

        ext_e_data = np.ascontiguousarray(ext_e_data, dtype=np.float64)
        ext_m_data = np.ascontiguousarray(ext_m_data, dtype=np.float64)

        self.raw[L].append(raw_data(T=T, M=ext_m_data, E=ext_e_data))

    def is_in(self,L:int,T:float,dT:float=10**-5)-> bool:
        """
        Check if the data for a specific system size and temperature is already stored.
        Args:
            L (int): The system size.
            T (float): The temperature.
            dT (float): The tolerance for temperature comparison. Default is 10^-5.
        Returns:
            bool: True if the data is already stored, False otherwise.
        """
        if L in self.raw.keys():
            for entry in self.raw[L]:
                if abs(entry.T - T) < dT:
                    return True
        return False
    
    def _sort(self):
        """
        Sort the data by temperature.
        """
        for L in self.raw.keys():
            self.raw[L].sort(key=lambda x: x.T)

    def see_loaded(self)-> None:
        """
        Print all the data stored in this object.
        """
        for L, data in self.raw.items():
            print(f"Data for L={L}:")
            for i,entry in enumerate(data):
                print(f"T={round(entry.T,3)} with {np.size(entry.M,axis=1)} measurements at index {i}")

    def delete(self,L:int,ind:int):
        """
        Delete a specific entry from the stored data.
        """
        if L in self.raw:
            if ind < len(self.raw[L]):
                del self.raw[L][ind]
            else:
                raise IndexError("Index out of range for energy data.")
        if L in self.raw:
            if ind < len(self.raw[L]):
                del self.raw[L][ind]
            else:
                raise IndexError("Index out of range for magnetization data.")
        raise ValueError("No data found for the specified system size.")




    def _ensemble_avg(self,f:Callable[[np.ndarray, np.ndarray, int, float], np.ndarray]):
        """
        Update self.processed, containing the average and error of the data based on the function f.
        Args:
            f (Callable): A function that takes magnetization, energy, system size, and temperature as arguments and returns a scalar value.
        >>> _ensemble_avg(lambda m, e, L, T: np.mean(m)) 
        # Computes the average magnetization.

        >>> _ensemble_avg(lambda m, e, L, T: np.mean(e*m))
        # Computes the average of energy times magnetization.

        >>> _ensemble_avg(lambda m, e, L, T: np.mean(np.abs(m)/L**2))
        # Computes the average of absolute magnetization divided by L^2.
        """
        self.processed = {}

        for L ,data in self.raw.items():
            if L not in self.processed:
                self.processed[L] = []
            for (t,m,e) in data:
                # Jackknife resampling
                Q_jack = []
                for i in range(np.size(m,axis=1)):
                    m_jack = np.delete(m, i, axis=1)
                    e_jack = np.delete(e, i, axis=1)
                    try:
                        Q_jack.append(f(m=m_jack, e=e_jack, L=L, T=t))
                    except:
                        raise RuntimeError("Error in function evaluation. Check the function f.")
                Q_jack = np.array(Q_jack)
                Q_avg = np.mean(Q_jack)
                Q_err = np.sqrt((len(Q_jack) - 1) / len(Q_jack)) * np.std(Q_jack, ddof=1)
                self.processed[L].append(result(T=t, avg=Q_avg, err=Q_err))

    def _str2lambda(self, expr:str)-> Callable:
        """
        convert a string to a lambda function
        example: <e*m> = np.mean(e*m,axis=0)
        """
        expr = expr.replace("<", "np.mean(")
        expr = expr.replace(">", ")")

        in_abs = False
        new_str = ""
        for letter in expr:
            if letter == "|":
                if in_abs:
                    new_str += ")"
                else:
                    new_str += "np.abs("
                in_abs = not in_abs
            else:
                new_str += letter

        return eval("lambda m,e,L,T: " + new_str, {"np": np})


    def plot(self,f:str | Callable,scale = False):
        """
        Plot the processed data based on the function f.
        Args:
            f (str | Callable): The function to plot the data. If a string is provided, it will be converted to a lambda function.
            scale (bool): If True, scale the data according to the best fit parameters.
        >>> data.plot(lambda m, e, L, T: np.mean(m/L**2))
        # Plots the average magnetization per site
        >>> data.plot('<m/L**2>') # this one will also work

        >>> data.plot('(<m**2>-<m**2>)/L**2/T') # susceptibility
        """
        if isinstance(f, str):
            f = self._str2lambda(f)
        self._ensemble_avg(f)
        for L, data in self.processed.items():
            if scale:
                T =( np.array([x.T for x in data]) - self.T_c)/self.T_c * L**self.b_best
                avg = np.array([x.avg for x in data])*L**self.a_best
                err = np.array([x.err for x in data])*L**self.a_best
                plt.errorbar(T, avg, yerr=err, label=f"L={L}", fmt='o', capsize=3,lw =1)
            else:
                T = [x.T for x in data]
                avg = [x.avg for x in data]
                err = [x.err for x in data]
                plt.errorbar(T, avg, yerr=err, label=f"L={L}", fmt='o', capsize=3,lw =1)
        plt.legend()
        return

    def fit_a(self,f:str|Callable):
        """
        Fit the data based on scaling hypothesis Q ~ L^{-a} f(t L^{1/nu}). So Q_max ~ L^{-a}.
        Args:
            f (Callable): The function to fit the data.
        >>> data.fit_a(data.logd(2)) # Built in dlog<m^k>/dt function.
        Fitted a: 1.009 ± 0.024
        >>> data.fit_a(data.binderd(2)) # Built in d B_2k/dt function.
        Fitted a: 0.953 ± 0.041
        """
        if isinstance(f, str):
            f = self._str2lambda(f)
        self._ensemble_avg(f)
        l_list = []
        y = []
        for L, data in self.processed.items():
            l_list.append(L)
            y.append(np.max([x.avg for x in data]))
        result = linregress(np.log(np.array(l_list)), np.log(np.array(y)))
        print(f"Fitted a: {result.slope:.3f} ± {result.stderr:.3f}")


    def fit_Tc(self,f):
        """
        Fit the critical temperature T_c based on existing nu. Fit the data by Tc(L) = T_c + A* L^(-1/nu)
        Args:
            f (Callable): The function to fit the data.
        """
        if isinstance(f, str):
            f = self._str2lambda(f)
        self._ensemble_avg(f)
        l_list = []
        y = []
        for L, data in self.processed.items():
            T = np.array([x.T for x in data])
            avg = np.array([x.avg for x in data])
            max_index = np.argmax(avg)
            l_list.append(L)
            y.append(T[max_index])
        result = linregress(np.array(l_list)**(-1/self.nu), np.array(y))
        print(f"Fitted T_c: {result.intercept:.3f} ± {result.intercept_stderr:.3f}")
    

