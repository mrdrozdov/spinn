import subprocess
import sys
import getpass

### Kill a job and its chain of dependents (as created by sbatch_submit).                                                                                                                      
### Usage: python sbatch_cancel.py [Name of first running job in chain]                                                                                                                        

CURRENT_JOB = sys.argv[1]
USER = getpass.getuser()

lines = subprocess.check_output(['squeue', '-u', USER, '-o', '"%.8A %.20E"'])
lines = lines.split('\n')
lines.sort()

to_kill = [CURRENT_JOB]

for line in lines:
        s = line.split()
        if (len(s) > 0) and (to_kill[-1] in s[2]):
                to_kill.append(s[1])

subprocess.call(['scancel'] + to_kill)

