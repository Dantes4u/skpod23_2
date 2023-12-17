import os
import time
from math import pi
gs = [128, 256]
ps = [1, 2, 4]
ts = [1, 2, 4, 8]
Ls = [1.0, pi]   
for g in gs:
    for p in ps:
        for t in ts:
            for L in Ls:
                for i in range(2):
                    file_out = f"results/{g}_{p}_{t}_{int(L)}_{i}.out"
                    os.system(
                        f"mpisubmit.pl -p {p} -t {t} --stdout {file_out} --stderr tmp.err mpi {L} {L} {L} {g} 0.5 50"
                    )
                    time.sleep(5)
