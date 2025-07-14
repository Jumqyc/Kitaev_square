from typing import NamedTuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

raw_data = NamedTuple('raw_data', [('T', float), ('Q', np.ndarray)])
result = NamedTuple('result', [("T",float),('avg', float), ('err', float)])

class fss:
    def __init__(self) -> None:
        self.e_raw :dict = {}
        self.m_raw :dict = {}
        self.processed = {}
        self.T_c = 0
        self.a_best = 0
        self.b_best = 0
        self.w = 2

    def add_raw_data(self,ext_e_data:np.ndarray,ext_m_data:np.ndarray,T:float,L:int):
        """
        Add raw data to this object. 
        Args:
            data (np.ndarray): The raw data to be added.
            T (float): The temperature at which the data was collected.
            L (int): The system size.
            type (str): The type of data, either 'e' for energy or 'm' for magnetization.
        """
        if len(ext_e_data) != len(ext_m_data):
            raise ValueError("The length of energy and magnetization data must be the same.")
        if L not in self.e_raw.keys():
            self.e_raw[L] = []
            self.m_raw[L] = []
        m_rtime = self._rtime(ext_m_data)
        e_rtime = self._rtime(ext_e_data)
        rtime = max(m_rtime, e_rtime)

        num = len(ext_e_data)

        ext_e_data = ext_e_data[:rtime * (num // rtime)]
        ext_m_data = ext_m_data[:rtime * (num // rtime)]

        ext_e_data = ext_e_data.reshape((rtime, num//rtime)) # reshape to (rtime, num/rtime). 
        # ALWAYS AVERAGE OVER AXIS 0!!!!!!!
        ext_m_data = ext_m_data.reshape((rtime, num//rtime))

        self.e_raw[L].append(raw_data(T=T, Q=ext_e_data))
        self.m_raw[L].append(raw_data(T=T, Q=ext_m_data))
    
    def _sort_data(self):
        """
        Sort the data by temperature.
        """
        for L in self.e_raw.keys():
            self.e_raw[L].sort(key=lambda x: x.T)
        for L in self.m_raw.keys():
            self.m_raw[L].sort(key=lambda x: x.T)

    def see_all_data(self):
        """
        print all the data stored in this object.
        """
        for L, data in self.e_raw.items():
            print(f"Data for L={L}:")
            for i,entry in enumerate(data):
                print(f"T={round(entry.T,3)} with {np.size(entry.Q,axis=0)} measurements at index {i}")

    def delete(self,L:int,ind:int):
        """
        Delete a specific entry from the stored data.
        """
        if L in self.e_raw:
            if ind < len(self.e_raw[L]):
                del self.e_raw[L][ind]
            else:
                raise IndexError("Index out of range for energy data.")
        if L in self.m_raw:
            if ind < len(self.m_raw[L]):
                del self.m_raw[L][ind]
            else:
                raise IndexError("Index out of range for magnetization data.")
        raise ValueError("No data found for the specified system size.")


    def _rtime(self, data:np.ndarray):
        N = len(data)
        data = data - np.mean(data)  # calculate variance around mean
        var = np.var(data, ddof=1)

        for search in range(100, N, N//100):
            acf = np.array([np.mean((data[:N-k]) * (data[k:])) for k in range(search)])
            if np.sum(acf < 0) > 0:
                break
        else:
            raise ValueError("No negative autocorrelation found in the data.")
        acf /= var
        k_max = np.argmax(acf < 0)
        if k_max == 0:
            k_max = len(acf)
        rtime = int(0.5 + np.sum(acf[1:k_max+1]))+1
        return 3*rtime 

    def ensemble_avg(self,f):
        """Update two dictionaries, containing the average and error of the energy and magnetization data."""
        self._sort_data()
        self.processed = {}

        for L in self.e_raw.keys():
            if L not in self.processed:
                self.processed[L] = []
            for i , m_raw in enumerate(self.m_raw[L]):
                e_raw = self.e_raw[L][i]
                Q = f(m = m_raw.Q,e = e_raw.Q, L= L,T = m_raw.T)
                self.processed[L].append(
                    result(T=m_raw.T, avg=np.mean(Q), err=np.std(Q, ddof=1)/np.sqrt(len(Q)))
                )

    def _interpolate(self):
        """for each system size L, create an interpolation function for the processed data."""
        output = {}
        for L , data in self.processed.items():
            output[L] = interp1d(
                [x.T for x in data],
                [x.avg for x in data],
                kind='linear',
                bounds_error=False,
                fill_value=np.nan
            )
        return output

    def determine_Tc(self,searchrange:tuple,plot: bool = True):
        """
        Determine critical temperature T_c using binder ratio
        """
        self._sort_data()
        self.ensemble_avg(lambda m,e,L,T: 1- (np.mean(m**4,axis=0) / (3 * np.mean(m**2,axis=0)**2)))
        interpolate_fun = self._interpolate()

        t_min = searchrange[0]
        t_max = searchrange[1]
        t = np.linspace(t_min, t_max, 1000)
        binder = np.zeros((len(interpolate_fun), len(t)))
        for i, L in enumerate(interpolate_fun.keys()):
            binder[i] = interpolate_fun[L](t)
            if plot:
                plt.plot(t, binder[i], label=f"L={L}")

        binderstd = np.nanstd(binder,axis=0)
        minind = np.argmin(binderstd,axis=0)
        print(f"Critical temperature T_c is {t[minind]}")
        self.T_c = t[minind]
        if plot:
            plt.plot(t, binderstd, label='Binder Standard Deviation')
            plt.show()

    def plot(self,f,scale = False):
        if f != None:
            self.ensemble_avg(f)
        for L, data in self.processed.items():
            if scale:
                T =( np.array([x.T for x in data]) - self.T_c)/self.T_c * L**self.a_best
                avg = np.array([x.avg for x in data])*L**self.b_best
                err = np.array([x.err for x in data])*L**self.b_best
                plt.errorbar(T, avg, yerr=err, label=f"L={L}", fmt='o', capsize=3,lw =3)
            else:
                T = [x.T for x in data]
                avg = [x.avg for x in data]
                err = [x.err for x in data]
                plt.errorbar(T, avg, yerr=err, label=f"L={L}", fmt='o', capsize=3,lw =3)
        return

    def S(self,a:float,b:float)-> np.float64:
        # step 1, scale the data by size to (x = (T - T_c)/T_c L^b, y = Q L^a,dy = dQ L^a)
        scale_data = {}
        for L in self.processed.keys():
            fix_L = []
            for d in self.processed[L]:
                fix_L.append(
                    result(T = (d.T - self.T_c)/self.T_c * L**a,
                           avg = d.avg * L**b,
                           err = d.err * L**b)
                            )
            scale_data[L] = fix_L
        # step 2, interpolate the data
        
        Y = []
        dY = []
        yij = []
        dyij = []
        mask = []

        for L1, fix_L1 in scale_data.items():
            for x,y,dy in fix_L1:               # for each data point
                wl = []
                xl = []
                yl = []
                for L2, fix_L2 in scale_data.items():
                    if L1 == L2:
                        continue      #compare with the other data sets with different sizes

                    idx = np.searchsorted([d.T for d in fix_L2], x, side='left')
                    if idx == 0 or idx >= len(fix_L2) - 1:
                        continue # skip if no point is found
                    wl.append(fix_L2[idx].err)
                    xl.append(fix_L2[idx].T)
                    yl.append(fix_L2[idx].avg)
                    wl.append(fix_L2[idx+1].err)
                    xl.append(fix_L2[idx+1].T)
                    yl.append(fix_L2[idx+1].avg)
                
                wl = 1/np.array(wl)**2
                xl = np.array(xl)
                yl = np.array(yl)

                K = np.nansum(wl)
                Kx = np.nansum(wl * xl)
                Ky = np.nansum(wl * yl)
                Kxx = np.nansum(wl * xl**2)
                Kxy = np.nansum(wl * xl * yl)
                Delta = K * Kxx - Kx**2

                Y.append((Kxx*Ky - Kx * Kxy)/ Delta + x*(K*Kxy- Kx*Ky)/Delta)
                dY.append((Kxx-2*x*Kx+x**2* K)/Delta)
                yij.append(y)
                dyij.append(dy)
                mask.append(np.exp(-x**2/(2*self.w**2)))  
        Y = np.array(Y)
        dY = np.array(dY)
        yij = np.array(yij)
        dyij = np.array(dyij)

        return np.nanmean(mask * (Y - yij)**2 / (dY + dyij**2))


    def fit_exponents(self, f):
        self._sort_data()

        self.ensemble_avg(f)
        S_best = np.inf
        a_best = 0
        b_best = 0
        for a in np.linspace(0.9, 1.2, 51):
            for b in np.linspace(0, 0.2, 51):
                S = self.S(a,b)
                if S < S_best:
                    S_best = S
                    a_best = a
                    b_best = b
                    print(f"Best fit: a = {a_best}, b = {b_best}, S = {S_best}")

        self.a_best = a_best
        self.b_best = b_best
