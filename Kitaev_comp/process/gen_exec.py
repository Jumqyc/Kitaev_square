import os


data_path = '/public/home/hkust_jwliu_1/yqianao/Kitaev_comp/dataAll'

for size in os.listdir(data_path):
    if size.startswith('N'):
        os.makedirs('./process_bash', exist_ok=True)
        with open(f'./process_bash/process_{size}.sh', 'w') as f:
            f.write(f"""#!/bin/bash\n
#SBATCH -n 1\n
#SBATCH -N 1\n
#SBATCH -t 0-60:00\n
#SBATCH --cpus-per-task=1\n
#SBATCH -p lzicnormal\n
#SBATCH --mem=10GB\n
#SBATCH -o process.out\n
#SBATCH -e process.err\n
# module load python/3.12\n      
cd /public/home/hkust_jwliu_1/yqianao/Kitaev_comp/process_script
numactl --interleave=all python3 process.py {size}""")


with open(f'./process_bash/exec_all.sh', 'w') as f:
    f.write("""#!/bin/bash
for job_script in exec_*.sh; do
    echo "Submitting job script: $job_script"
    sbatch "$job_script"
    sleep 1
done""")
