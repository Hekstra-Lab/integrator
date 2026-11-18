#!/user/bin/env python
"""
This script combines the 16 mtzs produced by laue-DIALS into four mtz files, one per timepoint, in preparation for reduction with careless. If any subfolders don't contain an integrated.mtz file, this script will print a slurm command that allows one to rerun the Laue-DIALS processing. 
"""


import pandas as pd
import reciprocalspaceship as rs
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import glob
import gemmi
import shutil
import sys

####SET VARIABLES HERE

times = ["off","50ns","100ns","200ns"]
#times = ["off","100ns","200ns"]
expts = ['c','d','e','f']
exptname='e35'
int_type='ld'
os.environ["int_type"] = int_type

gain = sys.argv[1]
gain_str = gain.replace(".",",")

####END SET VARIABLES

unit_cell = (65.261, 39.45, 38.961, 90.0, 117.456, 90.0) #this unit cell is taken from pdb 5E11. 
sg = 5 #C121 space group

base_dir = "."
int_dir = base_dir+f"/gain_{gain_str}_from_stills/"

os.chdir(int_dir)

int_dir = os.getcwd()

# Local deviation: pass d is processed with start angle 0 (the tutorial's pass-d
# workaround needs stills built with laue-dials 0.3.dev48; laue-dials issue #49).
# A few pass-d frames still refine badly, so drop every frame whose spot RMSD
# exceeds 2 px in any timepoint; off and on then keep the same frames.
import re
def bad_frames(pass_, cut=2.0):
    bad = set()
    for t in times:
        for ln in open(f"dials_files_{pass_}_{t}/rmsds.log"):
            m = re.match(r"\s*(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s*$", ln)
            if m and float(m.group(3)) > cut:
                bad.add(int(m.group(2)))
    return sorted(bad)
BAD_D = bad_frames("d")
print(f"pass d frames dropped (rmsd > 2 px in any timepoint): {BAD_D}")
mtz_dir =int_dir+f"/{int_type}_{gain_str}/"

try:
    os.mkdir(mtz_dir)
except:
    pass

mtzs = {}

# these are user input parameters. 
# phi_start and osc are dictionaries, with keys as passes,
#indicating start angle and oscillation angle. 
# spacer is a dictionary indicating the start frame of the pass. 

spacer={'c':0,'d':46,'e':45+47,'f':45+47+91}
phi_start={'c':0,'d':92,'e':181,'f':361.5}
osc={'c':2,'d':2,'e':2,'f':1}
sbatches=[]

def add_ohp(df, expt): #add one-hot pad metadata to distinguish pass identity. 
    for expt_ in expts:
        df[f"is_{expt_}"] = [expt==expt_]*len(df)
        df[f"is_{expt_}"] = df[f"is_{expt_}"].astype("MTZInt")
    return df 

continuing=True
for delay in times:
    for expt in expts:
        try:
            if expt=="c":
                mtzs[delay]=rs.read_mtz(f"dials_files_{expt}_{delay}/integrated.mtz") 
                mtzs[delay].cell=gemmi.UnitCell(*unit_cell)
                mtzs[delay] = add_ohp(mtzs[delay], expt)
            else:
                curr = rs.read_mtz(f"dials_files_{expt}_{delay}/integrated.mtz")
                curr.cell=gemmi.UnitCell(*unit_cell)
                
                #one-hot encoding of pass identity
                curr = add_ohp(curr, expt)

                
                ##we remove the reflections for images that are improperly indexed. These reflections 
                # are improperly indexed due to crystal slippage. 
                if expt == "d":
                    curr = curr[~curr["BATCH"].isin(BAD_D)]
                if expt == "e":
                    curr = curr[curr["BATCH"]<=85]
                if expt == "f":
                    curr = curr[curr["BATCH"]<=95]

                curr['BATCH']+=spacer[expt]
                mtzs[delay]=rs.concat([mtzs[delay],curr])
        except: #if the run fails, then we need to reprocess until success.
            if expt=="d":
                sbatches.append(f"\n sbatch one_pass-from_off-pass_d.sh {expt} {delay} {phi_start[expt]} {osc[expt]} {gain}")
            else:
                sbatches.append(f"\n sbatch one_pass-from_off.sh {expt} {delay} {phi_start[expt]} {osc[expt]} {gain}")
            continuing = False
        mtzs[delay]["is_c"]
            
if continuing:
    ds_times_with_sym = {}
    head = 'cdef'
    sg_str = 'P1'
    cb_op = "a+b,a-b,-c"

    for time in times:
        
        #we write the off, 50ns, 100ns, and 200ns mtzs, and then write the mtzs with shared miller indices. We additionally change the basis of each mtz file.  
        mtzs[time].write_mtz(f"{mtz_dir}{head}_{exptname}_{time}.mtz")
        merged = pd.merge(mtzs[time],mtzs['off'],left_index=True,right_index=True,how='inner')
        temp = mtzs[time].loc[merged.index.unique()]
        temp.write_mtz(f"{mtz_dir}{head}_{exptname}_{time}_shared_index.mtz")
        
        #we then write the reduced-symmetry space group mtzs. 
        temp.spacegroup = gemmi.SpaceGroup(sg_str)
        transform = gemmi.Op(cb_op)
        temp.cell = temp.cell.changed_basis_forward(transform, True)
        temp = temp.apply_symop(transform.inverse())
        temp.write_mtz(f"{mtz_dir}{head}_{exptname}_{time}_{sg_str}.mtz") 
    print("MTZs created.")
else:
    print("no MTZs created.")
    print(''.join(sbatches))