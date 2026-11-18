# The deposited laue-dials scripts, as they were run

Steps 7 to 10 of the PDZ2 chain -- the frame cuts, the eight-file careless
call, CCsym, and the weighted difference map -- are the rs-station
laue-dials tutorial's own scripts, not ours. Running them unchanged is the
point: integration is then the only thing that differs between our arms and
the published result.

They are vendored here because they existed in two places that both
disappear -- `laue_runs/pdz2_ref/scripts/` on netscratch, which has purged
this project once, and a Downloads folder.

## One copy, plus the deviation

The files here are the versions that **actually ran**, not the pristine
tutorial ones. Vendoring both would be two copies of nearly the same script,
which is the duplication this repository is trying to stop.

`upstream.patch` is the diff from the tutorial originals, so every deviation
is visible without a second copy. The originals are the `--- upstream/` side
of each hunk.

Four of the five scripts the chain uses were modified, and **one of those
changes is scientific rather than cosmetic**:

| script | changed | what |
|---|---|---|
| `expt_concat.py` | 17 lines | **Drops pass-d frames whose spot RMSD exceeds 2 px in any timepoint.** laue-dials issue #49 leaves pass d misindexed at the tutorial's start angle; we import it at start 0 instead, and a few frames still refine badly. Running the tutorial version merges DIFFERENT FRAMES and gives different numbers. |
| `careless-cdef-ohp-mlpw.sh` | 11 lines | SLURM partition, GPU type, conda to micromamba. **The careless flags are untouched**, so the merge is the published one. |
| `stats_for_production.sh` | 9 lines | environment and paths |
| `wdm.py` | 4 lines | paths |
| `custom_ccsym.py` | none | byte-identical to the tutorial |

So CCsym is the tutorial's own statistic unmodified, the careless merge uses
the tutorial's own flags, and the one place we depart is the pass-d frame
list -- which is recorded in `upstream.patch` and in `docs/methods.md`
rather than left to be discovered.

## Source

The rs-station laue-dials tutorials, PDZ2 / EF-X combine-passes tutorial.
A copy of the download is at `~/Downloads/laue_tutorials/pdz2/`, which also
holds `tutorial_combine_passes.ipynb` and the pass-d variant
`one_pass-from_off-pass_d.sh` that needs laue-dials 0.3.dev48.

Upstream carries no version marker in the files, so `upstream.patch` is the
record of what we started from. If the tutorial is re-downloaded and the
patch no longer applies, upstream has moved and the difference needs
reading before anything is re-run.
