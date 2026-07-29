# MFX101555026 integrator workflow notes

This folder contains the small, reproducible workflow files used for the MFX101555026 cctbx.xfel -> integrator prototype. The large raw data, generated shoebox datasets, training runs, prediction outputs, and diagnostic images stay in scratch and should not be committed to Git.

## Purpose

The goal of this workflow is to convert MFX cctbx.xfel integrated still-image outputs into an integrator-compatible dataset, then test training and prediction with MFX-friendly model/loss settings.

The current prototype supports:

- MFX `.refl/.expt` inputs from cctbx.xfel integration.
- Fixed-size 2D shoebox extraction, currently `1 x 25 x 25`.
- Pam/cctbx detector mask support.
- Metadata needed by the Wilson loss, including `d`, `H`, `K`, `L`, intensity, background, panel, and pixel-position fields.
- Metadata needed for MFX many-file write-back, including `refl_ids`, `image_num`, `image_id`, `n_images`, `reflection_id`, `source_refl`, and `source_expt`.
- Chunked MFX shoebox writing to reduce memory/file-pressure during extraction.
- MFX-compatible data loading from `.npy` files.
- `monochromatic_wilson` training with Normal observation likelihood for float/negative MFX/Jungfrau pixel values.
- Gamma, LogNormal, and FoldedNormal tests for the intensity posterior `qi`.
- Optional image-level Wilson B/G lookup-table training.
- MTZ writing from prediction parquet files.
- Integrated MFX `.refl` write-back through `integrator.predict --write-refl --mfx-writeback`.

## Big picture pipeline

```text
MFX cctbx.xfel integrated files
  idx-data_*_integrated.refl
  idx-data_*_integrated.expt
        |
        v
src/integrator/cli/make_shoeboxes.py --mfx
        |
        v
counts.npy, masks.npy, metadata.npy, dataset.yaml
        |
        v
RotationDataModule
        |
        v
integrator.train
        |
        v
integrator.predict
        |
        v
pred.parquet / MTZ outputs / MFX .refl write-back / diagnostics
        |
        v
cctbx.xfel.merge scaling + merging
        |
        v
merged MTZ outputs / CC1/2 diagnostics
Main code changes
1. src/integrator/cli/make_shoeboxes.py

Added and updated the MFX/cctbx.xfel shoebox extraction path.

MFX mode

The MFX mode reads cctbx.xfel integrated .refl/.expt pairs, loads raw detector pixels through the .expt imageset, crops fixed-size shoeboxes around predicted reflection centers, applies detector masks, and writes an integrator dataset.

Important options:

Option	Meaning
--mfx	Enables the MFX/cctbx.xfel extraction path.
--mfx-pattern	Selects integrated reflection files, for example idx-data_*_integrated.refl.
--start-file	Starts processing at an index in the sorted file list. Useful for chunks or smoke tests.
--max-files	Limits how many integrated files to process.
--detector-mask / --mask	Loads Pam/cctbx detector mask. True means good pixel.
--write-chunk-size	Number of shoeboxes to buffer before writing a temporary chunk.
--no-mask-overlap	Keeps the MFX extraction behavior used in the current tests.
--counts-dtype float32	Writes counts.npy as float32.
Important functions/helpers
Function	Purpose
run_mfx(args)	Main MFX extraction function. Finds .refl files, matches .expt files, loads reflections/experiments, gets raw panel pixels, crops shoeboxes, applies masks, and writes final dataset files.
_matching_mfx_expt(refl_path)	Finds the matching .expt file for one integrated .refl file.
_mfx_image_index_from_expt_json(expt_path)	Reads the .expt JSON-like text and extracts the single_file_indices value for metadata/debugging.
_mfx_d_spacing(experiment, miller_index)	Computes crystallographic d-spacing from the experiment crystal/unit cell and Miller index.
_crop_mfx_fixed_shoebox(...)	Crops a fixed-size shoebox from one detector panel, clips at panel edges, pads outside-panel regions, and creates a valid-pixel mask.
_raw_stats_from_memmap(...)	Computes raw mean/variance from counts.npy and masks.npy. This avoids Anscombe stats, which are unsafe for negative MFX pixels.
_mfx_metadata_rows_to_luis_dict(...)	Converts per-reflection metadata rows into Luis-style metadata.npy keys.
_save_concentration_from_memmap(...)	Computes and saves concentration.npy when stats are enabled and the environment supports the required dependencies.
merge_mfx_chunks(...)	Merges temporary chunk_00000, chunk_00001, ... folders into final counts.npy, masks.npy, and metadata.npy.
Chunked writing change

Old prototype:

.refl/.expt pair
-> load image + reflections
-> crop shoeboxes
-> append counts/masks/metadata to growing Python lists
-> repeat many files
-> save once at the end

Issue: the Python lists keep growing, which can increase memory pressure and can make larger MFX extraction less robust.

New chunked approach:

.refl/.expt pair
-> load image + reflections
-> crop shoeboxes
-> buffer up to write_chunk_size
-> flush chunk_00000, chunk_00001, ...
-> merge chunks
-> final counts.npy, masks.npy, metadata.npy

The final dataset format is unchanged. The change is only how intermediate shoeboxes are written during extraction.

Example smoke test result:

processed 25 MFX integrated file(s)
seen 12442 reflection(s)
saved 12442 fixed-size shoebox(es)
wrote chunk 0: 10000 shoeboxes
wrote chunk 1: 2442 shoeboxes
Metadata added for image-level Wilson and write-back

The MFX metadata now includes or preserves these important fields:

refl_ids
image_num
image_id
n_images
reflection_id
source_refl
source_expt

Purpose:

Field	Purpose
refl_ids	Global Integrator reflection/prediction id.
image_num	Original MFX image/file number.
image_id	Compact integer id from 0 to n_images - 1; used for per-image Wilson lookup.
n_images	Total number of unique image IDs.
reflection_id	Row index inside the original idx-data_*_integrated.refl file.
source_refl	Original .refl filename.
source_expt	Original .expt filename.

These fields make both image-level Wilson training and many-file MFX .refl write-back cleaner.

2. src/integrator/data_loaders/data_module.py

Updated the data loader so MFX .npy datasets can load through the existing integrator training path.

Important updates:

Area	Purpose
_load_shoebox_array(path, weights_only=True)	Loads counts.npy/masks.npy if available, otherwise falls back to torch.load for .pt files.
Metadata columns include d	Keeps d-spacing/resolution metadata available to the Wilson loss.
Metadata columns include image_id, image_num, n_images	Keeps image-level information available to the loss and predictions.
MFX variance key handling	Uses intensity.sum.variance when intensity.prf.variance is not present.
Transform control	Supports MFX-friendly transforms such as standardization, asinh, sqrt_squareplus, or none.
Flattened 2D shoebox handling	Supports arrays shaped (N, 625) for 25 x 25 shoeboxes.
is_test split support	Uses saved is_test metadata when available.

MFX pixels can be float and negative, so Anscombe-style transforms are unsafe. The MFX path uses raw/standardization-style handling or MFX-friendly transforms.

For image-level Wilson B/G, image_id and n_images should be integer tensors, because image_id indexes the learned embedding table.

3. src/integrator/model/loss/wilson_loss.py

Updated the Wilson loss to support MFX/Jungfrau pixel behavior, alternative qi posterior distributions, and optional image-level Wilson parameters.

Observation likelihoods

Added ObservationLikelihood, which supports:

Likelihood	Purpose
poisson	Preserves original count-like behavior.
normal	Used for MFX/Jungfrau float/negative pixels.
student_t	Tested as another robust option.

Important fields added to WilsonLoss.__init__:

observation_likelihood
init_obs_scale
student_t_df
eps
image_level_wilson
n_images

The direct Poisson pixel log-probability was replaced with:

ll = self.observation_model(rate, counts)

This keeps the same Wilson loss structure while allowing Normal or Student-t pixel likelihoods.

Monte Carlo KL fallback

The Wilson prior for intensity is still Gamma. For the Gamma baseline:

qi = Gamma
p_i = Gamma Wilson prior

PyTorch can compute the exact KL divergence.

For the new tests:

qi = LogNormal or FoldedNormal
p_i = Gamma Wilson prior

PyTorch does not have exact KL formulas for these pairs, so training initially failed with NotImplementedError.

Added helper:

_kl_divergence_with_mc_fallback(q, p, n_samples=8, eps=1.0e-8)

Logic:

Try PyTorch exact kl_divergence(q, p).
If it works, use it.
If it raises NotImplementedError, estimate KL by sampling from q.

Plain-English meaning:

qi is the model's predicted intensity distribution.
p_i is the Wilson Gamma prior.
The fallback samples possible intensity values from qi and checks whether those values are also reasonable under p_i.

This preserves the original Gamma behavior while allowing LogNormal and FoldedNormal qi to run.

4. src/integrator/model/loss/monochromatic_wilson_loss.py

Added optional image-level Wilson B/G behavior for monochromatic data.

Previous behavior:

one global B
one global G
shared across the whole dataset

New diagnostic behavior:

metadata["image_id"]
-> lookup B_image and G_image
-> compute Wilson prior rate per reflection

The image-level mode creates learned embedding tables:

raw_B_by_image = nn.Embedding(n_images, 1)
raw_G_by_image = nn.Embedding(n_images, 1)

During training:

image_id = metadata["image_id"].to(device).long()

B = softplus(raw_B_by_image(image_id)) + b_min
G = softplus(raw_G_by_image(image_id))

The prior rate is computed per reflection as:

tau_i = (1 / G_image) * exp(2 * B_image * s_sq)

where s_sq comes from the reflection resolution d.

The raw initialization is:

raw G = 0.0
raw B = 3.0

After softplus, the initial positive values are approximately:

G ≈ 0.69
B ≈ 3.05

All images start from the same initial B/G values, then training can learn image-specific corrections.

5. src/integrator/model/distributions/lognormal.py

Added LogNormal intensity posterior support.

Main pieces:

Item	Purpose
LogNormalDistribution(nn.Module)	Neural-network module that outputs a LogNormal distribution.
build_lognormal(**kwargs)	Registry factory function.
_lognormal_valid_args()	Defines accepted YAML args.
_reject_unknown_lognormal_args(kwargs)	Fails early on invalid YAML args.

Expected YAML args:

qi:
  name: lognormal
  args:
    eps: 1.0e-06
    scale_min: 1.0e-04
    in_features: 32

Do not use Gamma-only args such as reparameterization or k_min in the LogNormal block.

6. src/integrator/model/distributions/folded_normal.py

Added FoldedNormal intensity posterior support.

Main pieces:

Item	Purpose
FoldedNormal(Distribution)	Custom distribution implementing sample/rsample, log_prob, mean, and variance.
FoldedNormalDistribution(nn.Module)	Neural-network module that outputs a FoldedNormal distribution.
build_folded_normal(**kwargs)	Registry factory function.
_folded_normal_valid_args()	Defines accepted YAML args.
_reject_unknown_folded_normal_args(kwargs)	Fails early on invalid YAML args.

Why a custom distribution was needed:

Plain TransformedDistribution(Normal, AbsTransform) did not provide .mean.
It also failed during log_prob inside the MC KL fallback.
The custom class implements the FoldedNormal probability density directly:
If X ~ Normal(loc, scale), and Y = |X|,
then p_Y(y) = Normal(y | loc, scale) + Normal(-y | loc, scale), for y >= 0.

Expected YAML args:

qi:
  name: foldednormal
  args:
    eps: 1.0e-06
    scale_min: 1.0e-04
    in_features: 32
7. src/integrator/registry.py

Registered the new intensity posterior surrogate builders.

Added imports similar to:

from integrator.model.distributions.lognormal import build_lognormal
from integrator.model.distributions.folded_normal import build_folded_normal

Updated REGISTRY["surrogates"]:

"surrogates": {
    "gamma": build_gamma,
    "dirichlet": DirichletDistribution,
    "learned_basis_profile": ProfileSurrogate,
    "lognormal": build_lognormal,
    "foldednormal": build_folded_normal,
}

Important: LogNormal and FoldedNormal were added as surrogates, not priors. The Wilson prior p_i stays Gamma.

8. src/integrator/cli/predict.py

Added integrated MFX many-file .refl write-back support.

New command pattern:

dials.python -m integrator.cli.predict \
  --run-dir "$RUN" \
  --write-refl \
  --mfx-writeback \
  --original-refl-dir <original_cctbx_integrated_folder>

New arguments:

--mfx-writeback
--original-refl-dir

Purpose:

--write-refl
  enables prediction write-back into DIALS reflection tables

--mfx-writeback
  uses the MFX many-file write-back path

--original-refl-dir
  points to the folder containing the original cctbx integrated .refl/.expt files

Output path:

$RUN_DIR/predictions/mfx_refl_writeback/
9. src/integrator/io/pred_io.py

Updated prediction I/O so the prediction code can route to the MFX write-back path.

The MFX write-back path reads prediction parquet files, loads the dataset metadata, and writes predictions into copies of the original MFX .refl files.

10. src/integrator/io/refl_io.py

Added MFX many-file .refl write-back logic.

Mapping:

prediction parquet refl_ids
-> metadata.npy row
-> source_refl/source_expt or image_num + reflection_id
-> original idx-data_*_integrated.refl
-> correct row inside that .refl
-> write qi_mean / qi_var / qbg_mean
-> copy or symlink matching .expt unchanged

Values written:

qi_mean  -> intensity.sum.value and intensity.prf.value
qi_var   -> intensity.sum.variance and intensity.prf.variance
qbg_mean -> background.mean

This replaces the earlier temporary two-script workflow with an integrated integrator.predict --write-refl --mfx-writeback path.

11. src/integrator/io/mtz_io.py

Updated MTZ writing support.

For MTZ writing, the shoebox dataset’s dataset.yaml needs a crystal block. This is not part of the training YAML.

Example:

crystal:
  cell:
    - 53.814
    - 86.75
    - 140.465
    - 90.0
    - 90.0
    - 90.0
  space_group: P 21 21 21

Command:

python -m integrator.cli.predict --run-dir "$RUN_DIR" --write-mtz

The MTZ output and .refl write-back output are separate:

MTZ output:
  $RUN_DIR/predictions/epoch_XXXX/preds_epoch_XXXX.mtz

MFX .refl write-back output:
  $RUN_DIR/predictions/mfx_refl_writeback/
Workflow files in this folder

This workflow folder should contain only lightweight, reproducible files such as configs and scripts.

Suggested structure:

workflows/mfx101555026/
  README.md
  configs/
    mfx_normal_qpmean_config_log.yaml
    mfx_normal_qpmean_lognormal_qi.yaml
    mfx_normal_qpmean_foldednormal_qi.yaml
    mfx_normal_qpmean_foldednormal_qi_asinh.yaml
    mfx_normal_qpmean_foldednormal_qi_log_softplus.yaml
    mfx_normal_qpmean_foldednormal_qi_sqrt_squareplus.yaml
    mfx_normal_qpmean_run_paths.yaml
    mfx_smoke_train_with_d_125_normal_gpu.yaml
    mfx_smoke_train_with_d_125_studentt_gpu.yaml
    mfx_test_allruns_asinh_foldednormal_imagewilson.yaml
    mfx_test_allruns_sqrt_squareplus_foldednormal_imagewilson.yaml
  scripts/
    make_mfx_125_chunks_ulimit.sh
    make_mfx_next1000_chunks_ulimit.sh
    make_shoeboxes_allruns_024_rg101_16array.sh
    merge_allruns_024_rg101_shoebox_chunks.py
    merge_allruns_024_rg101_shoebox_chunks.sbatch
    merge_fastreader_chunks.py
    merge_mfx_shoebox_datasets.py
    diagnostics/
      compare_qi_25files.py
      plot_qi_meeting_25files.py
      inspect_mfx_shoeboxes.py
      inspect_qp_mean_profiles.py
Current configs
Shared settings

The 25-file posterior comparison used the same data/model settings and changed only qi.

Shared settings:

mode: monochromatic
loss:
  name: monochromatic_wilson
  args:
    observation_likelihood: normal
integrator:
  name: hierarchical
  args:
    encoder_out: 32
    mc_samples: 20
    data_dim: 2d
    d: 1
    h: 25
    w: 25
surrogates:
  qp:
    name: learned_basis_profile
  qbg:
    name: gamma
Gamma baseline
qi:
  name: gamma
  args:
    reparameterization: mean_fano
    eps: 1.0e-06
    k_min: 0.01
    in_features: 32
LogNormal test
qi:
  name: lognormal
  args:
    eps: 1.0e-06
    scale_min: 1.0e-04
    in_features: 32
FoldedNormal test
qi:
  name: foldednormal
  args:
    eps: 1.0e-06
    scale_min: 1.0e-04
    in_features: 32
Image-level Wilson B/G test
loss:
  name: monochromatic_wilson
  args:
    observation_likelihood: normal
    init_obs_scale: 10.0
    n_bins: 1
    image_level_wilson: true
    n_images: 10        # 35330 for full all-runs dataset
    lp_correction: false

Important prediction keys:

predict_keys:
  - refl_ids
  - is_test
  - qi_mean
  - qi_var
  - qp_mean
  - qbg_mean
  - qbg_var
  - intensity.sum.value
  - intensity.sum.variance
  - d
  - H
  - K
  - L
  - image_id
  - image_num
  - source_refl
  - source_expt

image_id is needed for the per-image Wilson lookup. source_refl and source_expt are useful for MFX many-file write-back.

Environment notes

Use separate environments for shoebox extraction / DIALS work and training / GPU prediction.

Shoebox extraction, MTZ writing, and .refl write-back environment

Use the cctbx/psana2/DIALS environment:

source /sdf/group/lcls/ds/tools/cctbx/psana2_setup.sh
export PYTHONPATH=/sdf/home/t/thaoh/s3df_practice/integrator/src:$PYTHONPATH
cd /sdf/home/t/thaoh/s3df_practice/integrator

Example CPU allocation:

srun --account lcls:prjlumine22 --partition milano -N 1 -c 32 --mem 256G -t 0-4:00 --pty bash -i

Use this environment for:

make_shoeboxes.py --mfx
dials.python -m integrator.cli.predict --write-refl --mfx-writeback
python -m integrator.cli.predict --write-mtz
cctbx.xfel.merge scale/merge PHILs
Training/prediction environment

Use integrator-cuda-dev:

srun --account lcls:prjlumine22 --partition ampere -N 1 -c 10 -G 1 --mem 96G -t 0-5:00 --pty bash -i
export MAMBA_ROOT_PREFIX=/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/micromamba_root
eval "$(/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/hewl_1118_tutorial/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate integrator-cuda-dev
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/sdf/home/t/thaoh/s3df_practice/integrator/src
cd /sdf/home/t/thaoh/s3df_practice/integrator

Check GPU:

nvidia-smi
python - <<'PY'
import torch
print(torch.cuda.is_available())
PY

Use this environment for:

integrator.train
integrator.predict
reading prediction parquet files
GPU training and inference
Example commands
Create 25-file MFX shoebox dataset with chunked writing
dials.python src/integrator/cli/make_shoeboxes.py \
  --mfx \
  --data-dir /sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/outputs/r0269/012_rg058/out \
  --out-dir /sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/mfx_shoebox_stream_test_25 \
  --mfx-pattern "idx-data_*_integrated.refl" \
  --start-file 0 \
  --max-files 25 \
  --detector-mask /sdf/data/lcls/ds/mfx/mfx101555026/results/pam/hot_lines_combined5.mask \
  --w 25 \
  --h 25 \
  --d 1 \
  --counts-dtype float32 \
  --write-chunk-size 10000 \
  --no-mask-overlap
Create 10-file all-runs smoke dataset
cd /sdf/home/t/thaoh/s3df_practice/integrator
source /sdf/group/lcls/ds/tools/cctbx/psana2_setup.sh
export PYTHONPATH=/sdf/home/t/thaoh/s3df_practice/integrator/src:$PYTHONPATH

BASE=/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx
DATA_DIR=$BASE/all_runs_269_289_no275_024_rg101_integrated
OUT_DIR=$BASE/mfx_shoebox_allruns_269_289_no275_024_rg101_test10
MASK=$BASE/masks/r0269_pam_stddev_border2_combined.mask

rm -rf "$OUT_DIR"

dials.python src/integrator/cli/make_shoeboxes.py \
  --mfx \
  --data-dir "$DATA_DIR" \
  --out-dir "$OUT_DIR" \
  --mfx-pattern "*_integrated.refl" \
  --start-file 0 \
  --max-files 10 \
  --detector-mask "$MASK" \
  --w 25 \
  --h 25 \
  --d 1 \
  --counts-dtype float32 \
  --write-chunk-size 10000 \
  --no-mask-overlap
Train Gamma baseline
integrator.train \
  --config /sdf/home/t/thaoh/s3df_practice/integrator/workflows/mfx101555026/configs/mfx_normal_qpmean_config_log.yaml \
  --log-dir /sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/runs/mfx_gamma_qi_25files
Train LogNormal qi
integrator.train \
  --config /sdf/home/t/thaoh/s3df_practice/integrator/workflows/mfx101555026/configs/mfx_normal_qpmean_lognormal_qi.yaml \
  --log-dir /sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/runs/mfx_lognormal_qi_25files_mc_kl
Train FoldedNormal qi
integrator.train \
  --config /sdf/home/t/thaoh/s3df_practice/integrator/workflows/mfx101555026/configs/mfx_normal_qpmean_foldednormal_qi.yaml \
  --log-dir /sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/runs/mfx_foldednormal_qi_25files_mc_kl
Train 10-file image-level Wilson B/G smoke test
integrator.train \
  --config /sdf/home/t/thaoh/s3df_practice/integrator/workflows/mfx101555026/configs/mfx_test_allruns_asinh_foldednormal_imagewilson.yaml \
  --log-dir /sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/runs/mfx_test10_asinh_foldednormal_imagewilson

Current smoke-test run:

/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/runs/mfx_test10_asinh_foldednormal_imagewilson/run_20260728-235319_08b4
Prediction

Use the newest successful run_* subfolder inside each log directory:

integrator.predict \
  --run-dir /sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/runs/<RUN_GROUP>/<RUN_SUBFOLDER>
MTZ writing

Make sure the dataset’s dataset.yaml has a crystal block.

Then run:

python -m integrator.cli.predict \
  --run-dir "$RUN_DIR" \
  --write-mtz

Current smoke-test MTZ output:

/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/runs/mfx_test10_asinh_foldednormal_imagewilson/run_20260728-235319_08b4/predictions/epoch_0024/preds_epoch_0024.mtz

Corresponding prediction parquet:

/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/runs/mfx_test10_asinh_foldednormal_imagewilson/run_20260728-235319_08b4/predictions/epoch_0024/preds_epoch_0024_rank=0_flush=000000.parquet
MFX .refl write-back
cd /sdf/home/t/thaoh/s3df_practice/integrator

source /sdf/group/lcls/ds/tools/cctbx/psana2_setup.sh
export PYTHONPATH=/sdf/home/t/thaoh/s3df_practice/integrator/src:$PYTHONPATH

RUN=/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/runs/mfx_test10_asinh_foldednormal_imagewilson/run_20260728-235319_08b4

dials.python -m integrator.cli.predict \
  --run-dir "$RUN" \
  --write-refl \
  --mfx-writeback \
  --original-refl-dir /sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/all_runs_269_289_no275_024_rg101_integrated

Expected output:

$RUN_DIR/predictions/mfx_refl_writeback/
Compare 25-file posterior outputs
python workflows/mfx101555026/scripts/diagnostics/compare_qi_25files.py
python workflows/mfx101555026/scripts/diagnostics/plot_qi_meeting_25files.py
Current 25-file smoke-test results

All three posterior tests produced prediction outputs.

Summary:

Model	qi_mean mean	qi_mean std	qi_mean max	qi_var mean	qi_var max	qbg_mean mean
Gamma	6.43	22.10	194	14.56	489	25.98
LogNormal	220.40	953.67	24,245	5,462	5.87M	14.83
FoldedNormal	7.34	23.42	214	174.84	20,176	26.06

Basic sanity checks:

n_rows = 30,400 for each prediction output
no NaNs in qi_mean / qi_var / qbg_mean
same metadata columns across runs

Current interpretation:

Gamma and FoldedNormal behave more similarly.
LogNormal runs successfully, but produces a much heavier right tail in qi_mean and qi_var.
FoldedNormal is currently closer to the Gamma baseline in this 25-file smoke test.

This is a debugging/validation result, not a final scientific conclusion.

Recent updates: all-runs MFX workflow

The workflow has now been extended beyond the earlier 25-file and single-run tests.

Current larger dataset target:

runs 269–289, excluding 275
trial/rungroup: 024_rg101

Staged all-runs integrated input folder:

/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/all_runs_269_289_no275_024_rg101_integrated

All-runs shoebox chunks:

/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/mfx_shoebox_allruns_269_289_no275_024_rg101_chunks

Intended merged all-runs dataset:

/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/mfx_shoebox_allruns_269_289_no275_024_rg101

All-runs metadata summary:

metadata rows:      115,146,952
unique image_num:   35,330
unique source_refl: 124,975
unique source_expt: 124,975
image_id range:     0–35,329

The main bottleneck is merging the full all-runs shoebox chunks. Estimated size:

counts.npy: ~288 GB
masks.npy:  ~72 GB
total:     ~360 GB plus metadata overhead

So the chunk merge needs a high-memory Slurm job or a more streaming-friendly layout.

10-file all-runs smoke-test results

Dataset path:

/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/mfx_shoebox_allruns_269_289_no275_024_rg101_test10

Result:

counts:        (8304, 625)
masks:         (8304, 625)
metadata rows: 8304
unique images: 10

The 10-file dataset was used to test:

make_shoeboxes
-> data loading
-> training
-> prediction
-> MTZ writing
-> MFX .refl write-back
-> image-level Wilson B/G path
Wilson B/G design note

The current per-image B/G lookup table should be treated as a diagnostic first implementation, not necessarily the final model.

Current diagnostic model:

per-image B
per-image G

Likely next model to test:

global B
per-image G

Reason:

G is strongly image-level because brightness/scale can change shot to shot.
B controls resolution-dependent decay and may need more shared signal to estimate reliably.

Recommended next addition:

track whether B/G actually move during training

Do not write learned B/G into metadata.npy. Metadata is static dataset information. Learned B/G depend on the run, checkpoint, epoch, and optimizer state.

Better output location:

run_dir/wilson_params/
  epoch_0005.parquet
  final_image_wilson_params.parquet

Suggested logged values:

wilson/B_mean
wilson/B_std
wilson/B_min
wilson/B_max
wilson/G_mean
wilson/G_std
wilson/G_min
wilson/G_max
wilson/B_grad_norm
wilson/G_grad_norm
wilson/B_delta_mean
wilson/G_delta_mean

Suggested exported table:

image_id
image_num
source_refl
B_image
G_image
Scaling / merging update

For the newer all-runs scaling tests, the CC1/2 spike/noisy behavior was fixed by setting:

filter.outlier.min_corr = 0.1

instead of:

filter.outlier.min_corr = -1.0

min_corr = -1.0 effectively disables the correlation-based outlier filter. min_corr = 0.1 removes low-correlation outliers and cleaned up the spike.

Also, d_min = 0 is not allowed in cctbx. It fails because the resolution limit must be greater than zero.

Instead, an optimistic diagnostic merge was run with:

merging.d_min = 1.5

The detector corner looked around:

~1.4–1.45 Å

The d_min = 1.5 merge showed that the signal drops before the detector edge, with CC1/2 near-zero or negative around:

~1.9–2.0 Å

So 1.5 Å is useful diagnostically, but likely too optimistic as a final cutoff.

Potential final cutoff range:

2.3–2.4 Å conservative
~2.2 Å optimistic
Visual diagnostics prepared

The meeting slide deck used as reference includes the following checks:

MFX shoebox extraction: old vs chunked flow.
Experiment setup: same data, three qi posterior choices.
25-file prediction summary.
25-file diagnostic plots for qi_mean, qi_var, and qi_mean vs intensity.sum.value.
Representative shoebox sanity check.
Top-20 strongest masked shoeboxes.
Input shoebox vs predicted profile qp_mean from the 1500-file Gamma run.
Wilson prior / intensity scaling architecture diagram:
previous global B/G model
current diagnostic per-image B/G lookup table
possible next global B + per-image G model
tracker/export plan for B/G diagnostics

The 1500-file Gamma qp_mean plot is currently the cleaner profile-quality visual. The 25-file Gamma/LogNormal/FoldedNormal runs are mainly for testing posterior behavior.

Files that should not be committed

Do not commit large data or run outputs:

runs/
outputs/
mfx_shoebox_*/
mfx_shoebox_chunks*/
diagnostics_qi_*/
diagnostics_qp_mean_*/
logs/
geom/
masks/
backup_*/
chunk_inputs/
*.npy
*.npz
*.parquet
*.ckpt
*.mtz
*.png
*.log
*.refl
*.expt

These should stay in scratch. Only lightweight configs, scripts, and notes should go into the repo.

Current status

The current MFX prototype can run:

MFX .refl/.expt
-> chunked shoebox extraction
-> integrator dataset
-> data loader
-> GPU training
-> prediction
-> posterior comparison diagnostics

The new qi posterior tests run with:

Gamma qi: exact KL
LogNormal qi: MC KL fallback vs Gamma Wilson prior
FoldedNormal qi: MC KL fallback vs Gamma Wilson prior

The newer all-runs workflow can now test:

combined all-runs cctbx integrated files
-> all-runs shoebox extraction
-> 10-file smoke dataset
-> image-level Wilson B/G training
-> GPU prediction
-> MTZ writing
-> integrated MFX .refl write-back
-> cctbx scaling/merging diagnostics

Current next questions / remaining items:

1. Finish/improve the full all-runs shoebox chunk merge.
2. Add Wilson B/G tracker callback.
3. Test global B + per-image G.
4. Run full all-runs Integrator training with n_images: 35330.
5. Run prediction on the full all-runs model.
6. Run MFX .refl write-back for the full all-runs predictions.
7. Scale/merge the full Integrator write-back output.
8. Compare against the all-runs cctbx baseline at matched d_min values.
9. Confirm which diagnostics are most useful before scaling further.
