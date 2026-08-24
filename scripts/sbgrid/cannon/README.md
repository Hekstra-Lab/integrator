# SBGrid dataset working directory

Copy this directory into the dataset folder, so the scripts that produced a
result sit beside it:

```bash
cp -r $INTEGRATOR_ROOT/scripts/sbgrid/cannon \
      /n/netscratch/hekstra_lab/Lab/laldama/sbgrid/<id>/scripts
```

`env.sh` is the only file to edit: `SBGRID_ID`, and `PDB_ID` when the dataset
names a deposition.

## Order

```bash
SBGRID_ID=821 PDB_ID=7LVC ./00_prepare.sh    # download, bundle, cards
./01_reference.sh                            # DIALS: import -> merge
./02_refine.sh                               # phenix.refine + anomalous peaks
./03_shoeboxes.sh                            # choose the window, cut them
./04_train.sh                                # the integrator
./02_refine.sh <integrator merged>.mtz       # the same refinement, its merge
```

`MAX_IMAGES=50 ./01_reference.sh` rehearses every DIALS step on a few frames
per sweep in minutes, which is worth doing before the full run.

## What each step leaves behind

| File | From | Read by |
| --- | --- | --- |
| `sbgrid_source.json` | download | provenance |
| `processing_hints.json` | depositor bundle | `dials_reference.py` |
| `reference_card.json`, `reference_merging_stats.csv` | the deposition | comparison |
| `dataset_card.json` | characterization | every later step |
| `dials_reference/merged.{mtz,html}` | DIALS | `plot_merging.py` |
| `dials_reference/merging_stats.csv` | the emitter | `plot_merging.py` |
| `*/refine/peaks.csv`, `refined_001.log` | phenix + peaks | `plot_peaks.py` |

Both arms write the same files, so the reference and the integrator land in
the same figures with no extra plotting.

## Why the steps are shaped this way

**The depositors' recipe wins over the image headers.** Where a processing
bundle exists it carries masks and geometry corrections the headers do not:
821's detector was raised between passes, so the header beam centre is right
for pass 1 and wrong for the others. Processing from headers alone would
index two thirds of that data incorrectly and report no error.

**The shoebox window is chosen, not configured.** `shoebox_size.py` reads the
bounding boxes DIALS actually used and takes the smallest odd window covering
a stated fraction of them, then reports what it clips and what it costs.

**R-free is generated once per dataset and copied.** Each arm generating its
own free set would hold out different reflections, and R-free would stop
being comparable between them.

**Anomalous is measured, not assumed.** The dataset card records a prior from
the deposited phasing method and the wavelength's distance from an absorption
edge; whether signal survives is what the peak search reports.
