#imports
import reciprocalspaceship as rs
from dxtbx.model.experiment_list import ExperimentListFactory
from dials.array_family import flex
import numpy as np
import pandas as pd


import gemmi
from cctbx import sgtbx
import sys

# Load data
#expts = ExperimentListFactory.from_json_file("../dials_files_c_off/poly_refined.expt")
#from IPython import embed; embed()
passes = ["c","d","e","f"]
timepoints = ["off","50ns","100ns","200ns"]

start_dir = sys.argv[1]
spot_df = []


reflections = flex.reflection_table().from_file(f"./{start_dir}/poly_refined.refl")
reflections = reflections.select(reflections.get_flags(reflections.flags.used_in_refinement))
from IPython import embed;embed()