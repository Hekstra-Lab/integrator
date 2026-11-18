# The hierarchical half of the 1000-frame ablation

`ablation_1118_1000.md` reports 18 arms crossing two integrator families
with five profile surrogates, and rows for both families -- `svae physical`
and `hier physical`, `svae learned` and `hier softmax_std`, and so on.

Only the **svae** configs were committed.
These eight are the hierarchical counterparts, recovered on 2026-11-18 from
`hier1000_cfgs/` on netscratch, which is purge-eligible, so half of a
published comparison was unreproducible from a clean checkout.

They are in a subdirectory because seven of the eight share a filename with
an svae config one level up and differ only in the `integrator.name` line.
Putting them side by side would have overwritten the svae set.

| file | `integrator` | `qp` | the run it produced |
|---|---|---|---|
| `softmax_std.yaml` | `hierarchical` | `learned_basis_profile` | `hier1000/softmax_std` |
| `softmax_std_expmean.yaml` | `hierarchical` | `learned_basis_profile` | `hier1000/softmax_std_expmean` |
| `entmax_std.yaml` | `hierarchical` | `learned_basis_profile` | `hier1000/entmax_std` |
| `softmax_geomboth.yaml` | `hierarchical` | `geometric_basis_profile` | `hier1000/softmax_geomboth` |
| `entmax_geomboth.yaml` | `hierarchical` | `geometric_basis_profile` | `hier1000/entmax_geomboth` |
| `fixed.yaml` | `hierarchical` | `fixed_basis_profile` | `hier1000/fixed` |
| `physical.yaml` | `hierarchical` | `physical_profile` | `hier1000/physical` |
| `shapelet.yaml` | `hierarchical` | `shapelet_profile` | `hier1000/shapelet` |

The svae runs of the same ablation are under `full1000/` on netscratch, from
the configs one level up.
