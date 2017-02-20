import subprocess
import sys
import getpass

### Kill a job and its chain of dependents (as created by sbatch_submit).
### Usage: python sbatch_cancel.py [Name of first running job in chain]

CURRENT_JOB = sys.argv[1]
USER = getpass.getuser()
	
lines = subprocess.call(['squeue', '-u', USER, '-o', '"%.8A %.4C %.10m %.20E"'])

to_kill = [CURRENT_JOB]

lines.sort()
for line in lines:
	s = line.split()
	if to_kill[-1] in s[1]:
		to_kill.append(s[0])

print subprocess.call(['scancel'] + to_kill)

