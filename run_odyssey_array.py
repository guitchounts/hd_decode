"""
Create sbatch submission file with proper parameters and submit the array job 

"""


import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('-cmd', type=str)
parser.add_argument('-days', type=int, default=0)
parser.add_argument('-hours', type=int, default=2)
parser.add_argument('-mem', type=int, default=500)
parser.add_argument('-cores', type=int, default=1)
parser.add_argument('-partition', type=str, default='shared', choices=['shared', 'cox','bigmem','serial_requeue','gpu','unrestricted'])
parser.add_argument('-mail', action='store_true')
parser.add_argument('-expt', type=str)
parser.add_argument('-env', type=str,default='ephys')
parser.add_argument('-mock', action='store_true')

parser.add_argument('-paramfile', type=argparse.FileType('r'))

settings = parser.parse_args(); 

N = sum(1 for line in settings.paramfile)

fo = open("%s.sbatch" % settings.expt, "w")

jobarrayname = "d_%s_%d" % (settings.expt, N)
jobname = "d_%s_%%a_%d" % (settings.expt, N)
fo.write("#!/bin/bash\n\n")
fo.write("#SBATCH --job-name=%s\n" % jobarrayname)
fo.write("#SBATCH --output=/n/holyscratch01/cox_lab/Users/guitchounts/ephys/hd_decoding_results/expt_%s/logs/%s.out\n" % (settings.expt, jobname))
fo.write("#SBATCH --error=/n/holyscratch01/cox_lab/Users/guitchounts/ephys/hd_decoding_results/expt_%s/logs/%s.err\n" % (settings.expt,jobname))
fo.write("#SBATCH -t %d-%d:00\n"% (settings.days,settings.hours))
fo.write("#SBATCH -p %s\n" % settings.partition)
fo.write("#SBATCH -n %d\n" % settings.cores)
fo.write("#SBATCH -N 1\n")
fo.write("#SBATCH --mem=%d\n" % settings.mem)
#fo.write("#SBATCH --requeue ")
if 'gpu' in settings.partition:
    fo.write("#SBATCH --gres=gpu:1\n")

if settings.mail:
    fo.write('#SBATCH --mail-type=FAIL\n')
    fo.write('#SBATCH --mail-user=guitchounts@fas.harvard.edu\n')


fo.write("module load Anaconda3/5.0.1-fasrc02 cuda/10.0.130-fasrc01 cudnn/7.4.1.5_cuda10.0-fasrc01 \n") # module load Anaconda3/5.0.1-fasrc02; source activate ephys
fo.write("source activate %s \n" % settings.env)
fo.write("\ncd %s\n" %  os.getcwd())
fo.write("python %s -paramfile %s -line ${SLURM_ARRAY_TASK_ID}\n" % (settings.cmd,settings.paramfile.name))

fo.close()

if not settings.mock:
    os.system("sbatch --array=0-%d %s"% (N-1,fo.name))


