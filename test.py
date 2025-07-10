import Spin
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle as pkl

def FT_record(record):
    """Discrete Fourier Transform of the record."""
    L = record.shape[1]
    FT = np.fft.fft2(record, axes=(1, 2),norm="ortho")
    FT = np.roll(FT, shift=L//2, axis=1)
    FT = np.roll(FT, shift=L//2, axis=2)
    FT = np.abs(FT)
    FT = np.average(FT, axis=0)
    return FT

t = time.time()
model = Spin.Spin(64)
model.set_parameters(0.1, 0, 1 ,0.5 , 0,- 0.3)
model.run(step = 1,spacing = 10000)
print(f"Time taken: for {100*1000*64*64} calculations:", time.time() - t)
Spin_record = model.get_saving()
FT_spin = FT_record(Spin_record)
fig,axs = plt.subplots(1,2)
Spin_record = np.average(Spin_record, axis=0)
axs[0].quiver(FT_spin[...,0], FT_spin[...,1],FT_spin[...,2])
axs[1].quiver(Spin_record[...,0], Spin_record[...,1],Spin_record[...,2],scale = 50)
axs[0].set_title("Fourier Transform of Spin Record")
axs[1].set_title("Spin Record")
axs[0].set_xlabel("kx")
axs[0].set_ylabel("ky")
axs[1].set_xlabel("x")
axs[1].set_ylabel("y")
axs[0].set_aspect('equal')
axs[1].set_aspect('equal')

plt.tight_layout()
plt.show()