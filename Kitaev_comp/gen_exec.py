from pathlib import Path
from decimal import Decimal, getcontext
import shutil
import numpy as np
import pandas as pd
import os
#this script creates slurm bash files for exec_noChecking_s.py
def format_using_decimal(value, precision=6):
    # Set the precision higher to ensure correct conversion
    getcontext().prec = precision + 2
    # Convert the float to a Decimal with exact precision
    decimal_value = Decimal(str(value))
    # Normalize to remove trailing zeros
    formatted_value = decimal_value.quantize(Decimal(1)) if decimal_value == decimal_value.to_integral() else decimal_value.normalize()
    return str(formatted_value)


outPath="./bashFiles/"
if os.path.isdir(outPath):
    shutil.rmtree(outPath)

Path(outPath).mkdir(exist_ok=True,parents=True)
num_parallel=24
which_row=1
chunk_size = 100
TVals= np.arange(0.1,0.5,0.025).round(4).tolist()
N_list = [8,16,32,64]

chunks = [TVals[i:i + chunk_size] for i in range(0, len(TVals), chunk_size)]

def contents_to_bash(chk_ind,T_ind,chunks,N):
    TStr=format_using_decimal(chunks[chk_ind][T_ind])
    contents=[
        "#!/bin/bash\n",
        "#SBATCH -n 1\n",
        "#SBATCH -N 1\n",
        "#SBATCH -t 0-60:00\n",
        f"#SBATCH --cpus-per-task={num_parallel}\n",
        "#SBATCH -p lzicnormal\n",
        "#SBATCH --mem=10GB\n",
        f"#SBATCH -o out_exec_T{TStr}.out\n",
        f"#SBATCH -e out_exec_T{TStr}.err\n",
        "cd /public/home/hkust_jwliu_1/yqianao/Kitaev_comp\n",
        f"python3 -u launch_one_run.py ./dataAll/N{N}/row{which_row}/T{TStr}/run_T{TStr}.mc.conf\n",
        f"numactl --interleave=all  ./run_mc ./dataAll/N{N}/row{which_row}/T{TStr}/cppIn.txt\n"
    ]

    out_chunk=outPath+f"/chunk{chk_ind}/"
    Path(out_chunk).mkdir(exist_ok=True,parents=True)
    outBashName=out_chunk+f"/exec_{n}_{TStr}.sh"
    with open(outBashName,"w+") as fptr:
        fptr.writelines(contents)

for n in N_list:
    for chk_ind in range(0,len(chunks)):
        for T_ind in range(0,len(chunks[chk_ind])):
            contents_to_bash(chk_ind,T_ind,chunks,n)