#!/user/bin/env python
"""
Compute CCsym from careless output. The plot parameters, timepoints, and mtz file numberings have been hardcoded for the PDZ2 EFX dataset. 

Note: This method currently assumes that careless has been called using
both the reduced-symmetry and the parent spacegroup
"""
import argparse
import numpy as np
import reciprocalspaceship as rs
import pandas as pd
import gemmi

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns


def parse_arguments():
    """Parse commandline arguments"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__
    )

    # Required arguments
    parser.add_argument(
        "mtz",
        nargs="+",
        help="MTZs containing crossvalidation data from careless",
    )
    parser.add_argument(
        "--op",
        required=True,
        help=(
            "Symmetry operation to use to compute internal difference map. "
            "Can be given as ISYM if used with a `spacegroup` argument"
        ),
    )
    
    parser.add_argument(
        "--cb_op",
        default="x,y,z",
        help=(
            "change-of-basis operation that reindexes low-symmetry miller indices to"
            "high-symmetry miller indices." 
        ),
    )

    # Optional arguments
    parser.add_argument(
        "-sg",
        "--spacegroup",
        help=(
            "Spacegroup to use for symmetry operation "
            "(only necessary if `op` specifies an ISYM)."
        ),
    )
    parser.add_argument(
        "-m",
        "--method",
        default="spearman",
        choices=["spearman", "pearson","weighted_pearson"],
        help=("Method for computing correlation coefficient (spearman or pearson or weighted pearson)"),
    )
    parser.add_argument(
        "--mod2",
        action="store_true",
        help=("Use id mod 2 to assign delays (use when employing spacegroup hack)"),
    )
    
    parser.add_argument(
        "-p",
        "--plotprefix",
        default="ccsym",
        help=("Prefix for filename of plot")
    )
    parser.add_argument(
        "-b",
        "--bins",
        default=10,
        type=int,
        help=("Number of resolution bins to use, the default is 10."),
    )

    return parser.parse_args()


def make_halves_ccsym(mtz, op, cb_op, bins=10):
    """Construct half-datasets for computing CCsym"""
    half1 = mtz.loc[mtz.half == 0].copy()
    half2 = mtz.loc[mtz.half == 1].copy()
    
    transformed_op = gemmi.Op(cb_op).inverse()*op*gemmi.Op(cb_op) #change matrix basis
    print(transformed_op)
    temp1 = half1.merge(
        half1.apply_symop(transformed_op).hkl_to_asu(),
        on=["H", "K", "L", "repeat"],
        suffixes=("1", "2"),
    )
    ## THERE IS A BUG HERE RELATING TO REPEATS AND EXPANDING TO P1. TRY TO REWRITE THE STRUCTURE OF THIS EVENTUALLY
    temp2 = half2.merge(
        half2.apply_symop(transformed_op).hkl_to_asu(),
        on=["H", "K", "L", "repeat"],
        suffixes=("1", "2"),
    )

    temp1["DF"] = temp1["F1"] - temp1["F2"]
    temp1["Sig2DF"] = temp1["SigF1"]**2+temp1["SigF2"]**2 #squared error
    temp2["DF"] = temp2["F1"] - temp2["F2"]
    temp2["Sig2DF"] = temp2["SigF1"]**2+temp2["SigF2"]**2

    temp = temp1[["DF", "Sig2DF", "repeat"]].merge(
        temp2[["DF", "Sig2DF","repeat"]], on=["H", "K", "L", "repeat"], suffixes=("1", "2")
    )

    temp, labels = temp.assign_resolution_bins(bins)

    return temp, labels

#I use the inverse variance weights and I use the DF1 variances as opposed to the DF2 variances. 
def compute_weighted_pearson(sub_df):
    return rs.utils.stats.weighted_pearsonr(sub_df["DF1"].to_numpy(),sub_df["DF2"].to_numpy(),
                                            np.reciprocal(sub_df["Sig2DF1"]).to_numpy())

def analyze_ccsym_mtz(
    mtzpath, op, cb_op, bins=10, return_labels=True, method="spearman", mod2=False
):
    """Compute CCsym from 2-fold cross-validation"""

    mtz = rs.read_mtz(mtzpath)
    m, labels = make_halves_ccsym(mtz, op, cb_op, bins)

    grouper = m.groupby(["bin", "repeat"])[["DF1", "DF2"]]
    if method == "weighted_pearson":
        result = grouper.apply(compute_weighted_pearson).to_frame().reset_index()
        result = result.rename(columns={0:('DF1', 'DF2')})
    else:
        result = (
            grouper.corr(method=method).unstack()[("DF1", "DF2")].to_frame().reset_index()
        )

    if mod2:
        result["delay"] = np.floor(int(mtzpath[-5]) / 2)
    else: 
        if int(mtzpath[-5]) == 5:
            result["delay"] = '50ns'
        elif int(mtzpath[-5]) == 7:
            result["delay"] = '200ns'
        elif int(mtzpath[-5]) == 4:
            result["delay"] = 'off'
        elif int(mtzpath[-5]) == 6:
            result["delay"] = '100ns'
        # if int(mtzpath[-5]) == 1:
        #     result["delay"] = 'off'
        # elif int(mtzpath[-5]) == 2:
        #     result["delay"] = '50ns'
        # elif int(mtzpath[-5]) == 3:
        #     result["delay"] = '100ns'
        # elif int(mtzpath[-5]) == 4:
        #     result["delay"] = '200ns'
            
    if return_labels:
        return result, labels
    else:
        return result


def main():

    # Parse commandline arguments
    args = parse_arguments()

    # Get symmetry operation
    try:
        isym = int(args.op)
        sg = gemmi.SpaceGroup(args.spacegroup)
        op = sg.operations().sym_ops[isym]
        cb_op = args.cb_op
    except ValueError:
        op = gemmi.Op(args.op)

    results = []
    results_overall = []
    labels = None
    for m in args.mtz:
        
        result = analyze_ccsym_mtz(m, op, cb_op, bins=args.bins, method=args.method, mod2=args.mod2)
        if result is None:
            continue
        else:
            result[0]["bin"]=result[0]["bin"]*2
            results.append(result[0])
            labels = result[1]
            
    labels = [labels[0].split('-')[0].strip()] + [i.split('-')[1].strip() for i in labels]
    
    results = rs.concat(results, check_isomorphous=False)
    
    results = results.reset_index(drop=True)
    results["bin"] = results["bin"].astype(int)
    results["CCsym"] = results[("DF1", "DF2")].astype(float)
    
    results.drop(columns=[("DF1", "DF2")], inplace=True)

    results.to_csv(f"{args.plotprefix}.csv")
    
    new_rc_params = {
        'text.usetex': False,
        "svg.fonttype": 'none',
        "font.size" : 18,
        #"font.family" : "sans-serif",
        #"font.sans-serif" : ["Nimbus Sans"],
    }
    matplotlib.rcParams.update(new_rc_params)
    fig,ax=plt.subplots(figsize=[8,5])
    sns.lineplot(
        data=results.dropna(), x="bin", y="CCsym", hue="delay", ci="sd",ax=ax, palette=["gray","#503050","#702070","#A000A0"]
    )
        
    plt.title(r"CCsym of PDZ2")
    plt.xticks(2*(np.arange(len(labels))-0.5), labels, rotation=45, ha="right", rotation_mode="anchor")
    plt.xlabel(f"resolution bin (Å)")
    plt.ylabel(r"CCsym " + f"({args.method})")
    plt.legend(loc="upper right")
    plt.xlim([-1.25,19.25])
    plt.ylim([-0.07,0.45])

    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{args.plotprefix}.svg", dpi=600)


if __name__ == "__main__":
    main()