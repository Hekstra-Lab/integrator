#!/user/bin/env python
"""
Compute a weighted difference map from the careless output. The basis-change operation and the mtz file numberings have been hardcoded for the PDZ2 EFX dataset.
""" 


import reciprocalspaceship as rs
import numpy as np
import os
import gemmi
import sys
#import pandas as pd

expt = 'e35'

######
#arguments that go into the rs diffmap computation
######

gain = sys.argv[1] 
gain_str = gain.replace(".",",")
base_dir = f'gain_{gain_str}_from_stills/'
merged_mtz_dir=f"{base_dir}/careless-cdef-{gain_str}-ohp-mlpw"
sg_str="P1"
cb_op = 'a+b,a-b,-c'
#cb_op = 'x,y,z'

def random_fill(row):
    #row is an R-free-flag entry
    if np.isnan(row):
        return 0
        #return np.random.choice([0, 1], p=[0.92, 0.08])
    else:
        return row
    
def expand_Rfree(mtz_name, sg_str, cb_op):
    mtz = rs.read_mtz(mtz_name)
    temp = mtz.expand_to_p1()
    temp.spacegroup = gemmi.SpaceGroup(sg_str)
    transform = gemmi.Op(cb_op)#.inverse()
    #transform = temp.spacegroup.centred_to_primitive()
    temp.cell = temp.cell.changed_basis_forward(transform, True)
    print(temp.cell)
    temp = temp.apply_symop(transform.inverse()).hkl_to_asu()
    return temp

def transfer_phases(small_mtz, big_mtz, sg_str, cb_op):
    new_asu = expand_Rfree(small_mtz, sg_str, cb_op)
    mtz_r3h = rs.read_mtz(big_mtz)
    # 5E1Y's cell differs slightly from the merged data's, so rs calls them
    # non-isomorphous; phases transfer by Miller index, so relax the check.
    new = mtz_r3h.merge(new_asu[["PHIC"]],on=["H","K","L"],how="left",check_isomorphous=False)
    new["PHIC"]=new["PHIC"].apply(random_fill)
    new["PHIC"]=new["PHIC"].astype("Phase")
    return new

a = transfer_phases(base_dir+"../5e1y_phases.mtz",
                f"{merged_mtz_dir}/merged_{expt}_7.mtz",sg_str,cb_op,
               ).write_mtz(f"{merged_mtz_dir}/merged_{expt}_7_phases.mtz")



cmd = f"""rs.scaleit -r {merged_mtz_dir}/merged_{expt}_4.mtz F SigF \
-i {merged_mtz_dir}/merged_{expt}_7.mtz F SigF \
-o {merged_mtz_dir}/merged_{expt}_7_scaled.mtz"""

os.system(f"source /n/hekstra_lab_tier0/Lab/garden/ccp4/ccp4-7.1/bin/ccp4.setup-sh; {cmd}")

cmd = f"""rs.diffmap -on {merged_mtz_dir}/merged_{expt}_7_scaled.mtz FPH1 SIGFPH1 \
-off {merged_mtz_dir}/merged_{expt}_4.mtz F SigF \
-r {merged_mtz_dir}/merged_{expt}_7_phases.mtz PHIC \
-a 0.05 -o {base_dir}wdm.mtz"""

os.system(cmd)

new = rs.read_mtz(f"{base_dir}wdm.mtz").compute_dHKL()
op = gemmi.Op("-x,y,-z")
transformed_op = gemmi.Op(cb_op).inverse()*op*gemmi.Op(cb_op) #change matrix basis
new = new.apply_symop(transformed_op).hkl_to_asu()
new.loc[(new.dHKL>1.8)*(new.dHKL<5)].write_mtz(f"{base_dir}wdm_rescut.mtz")

# cmd = f"""rs.internal_diffmap -i {merged_mtz_dir}/merged_{expt}_7_phases.mtz \
# F SigF -r {merged_mtz_dir}/merged_{expt}_7_phases.mtz PHIC -op 1 -sg 5 --cb_op={cb_op} \
# -a 0.05 -d 5 -o {merged_mtz_dir}/idm_{gain_str}.mtz"""
# os.system(cmd)
