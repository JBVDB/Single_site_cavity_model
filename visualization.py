# !conda install -c salilab dssp -y

import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from typing import Union, Dict

import Bio.PDB
import Bio.PDB.Polypeptide
from Bio.PDB.DSSP import dssp_dict_from_pdb_file, residue_max_acc

def get_dssp_df_on_file(pdb_file):
    """
    Run DSSP directly on a structure file with the Biopython method
    Bio.PDB.DSSP.dssp_dict_from_pdb_file.

    Args:
        pdb_file: Path to PDB file
    Returns:
        DataFrame: DSSP results summarized
    """
    try:
        d = dssp_dict_from_pdb_file(pdb_file)
        
    except:
        print(Exception('DSSP failed to produce an output'))
        return pd.DataFrame()

    appender = []   
    for k in d[1]:
        to_append = []
        y = d[0][k]
        chain = k[0]

        residue = k[1]
        het = residue[0]
        resnum = residue[1]
        icode = residue[2]

        to_append.extend([chain, resnum, icode])
        to_append.extend(y)
        appender.append(to_append)

    cols = ['chain', 'resnum', 'icode',
            'aa', 'ss', 'exposure_rsa', 'phi', 'psi', 'dssp_index',
            'NH_O_1_relidx', 'NH_O_1_energy', 'O_NH_1_relidx',
            'O_NH_1_energy', 'NH_O_2_relidx', 'NH_O_2_energy',
            'O_NH_2_relidx', 'O_NH_2_energy']
    
    df = pd.DataFrame.from_records(appender, columns=cols)

    # Adding additional columns.
    df = df[df['aa'].isin(list(Bio.PDB.Polypeptide.aa1))]
    df['aa_three'] = df['aa'].apply(Bio.PDB.Polypeptide.one_to_three)

    # Compute relative accessible surface area with Sander method (prop of RSA).
    df['max_acc'] = df['aa_three'].map(residue_max_acc['Sander'].get)
    df[['exposure_rsa', 'max_acc']] = df[['exposure_rsa', 'max_acc']].astype(float)
    df['exposure_asa'] = df['exposure_rsa'] * df['max_acc']

    df['rel_acc'] = df['exposure_rsa']/df['max_acc']

    # df.to_csv(f"results/pdb_file_dssp_res")

    return df


def classify_res_by_prot_region(dset_rel_acc: pd.Series,
                                low_thr: float=0.1,
                                up_thr: float=0.4):
    """Defining cutoffs for classification of the positions into core,
     boundary and surface.
     """

    classification = []
    for i in dset_rel_acc:
        if i < low_thr:
            classification.append('core')
        if low_thr <= i < up_thr:
            classification.append('boundary')
        if i >= up_thr:
            classification.append('surface') 
    return classification


def dataset_distribution(df: pd.DataFrame, ddg_cols, titles):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4),
                            constrained_layout=True, sharey=True)

    for i, (ax, ddg_col, title) in enumerate(zip(axes, ddg_cols, titles)):
        x = pd.Series(df[ddg_col], name='\u0394\u0394G (kcal/mol)')
        sns.histplot(x, kde=False, bins=25, stat="probability",
                    color="grey", edgecolor='black',
                    ax=ax,
                    zorder=10).set(ylabel="Fraction of Total Library")
        ax.axvline(x.median(), color='r',
                   linestyle='dashed', linewidth=1,
                   label="median")
        ax.grid(axis="y", linestyle='dashed', alpha=0.7)
        legend = ax.legend()
        frame = legend.get_frame()
        frame.set_facecolor('white')
        frame.set_edgecolor('black')
        ax.set_title(title)

    plt.savefig(f"results/stability_distribution_{df.pdbid.iloc[0]}.png",
                dpi=200, bbox_inches = "tight")
    plt.show()


def heatmap_stability_landscape(df: pd.DataFrame, ddg_cols, titles):
    # Specify order of target residues for the heatmap.
    aa_order_heatmap = list("DEHKRAFILMVYWNQSTGPC")
    dict_aa_order = {aa: i for i, aa in enumerate(aa_order_heatmap)}

    # Order values for the heatmap.
    df = df.sort_values(
    by="to", key=lambda x: x.map(dict_aa_order))

    # Adjust xlabels for homogeneity.
    def get_xlabels(resnum, aa):
        if resnum < 9:
            return f" {resnum} {aa}"
        else:
            return f"{resnum} {aa}"

    wt_seq = df[["resnum", "aa"]].drop_duplicates(
        ).apply(lambda x: get_xlabels(x['resnum'], x['aa']), axis=1
        ).to_numpy()

    # Get the indices of the core residues.
    list_core_pos = np.array(
        df[df["classification"] == "core"]["resnum"].unique() - 1
        )

    # Mask of indices not in core.
    mask = np.ones_like(wt_seq, dtype=bool)
    mask[list_core_pos] = False

    xrange_ = np.arange(df.resnum.min(), df.resnum.max()+1)

    for ddg_col, title in zip(ddg_cols, titles):
        trace = [go.Heatmap(
            x = df['resnum'],
            y = [df["aa_feature"], df['to']],
            z = df[ddg_col],
            type = 'heatmap',
            colorscale = 'RdBu_r',
            zmid=0,
            colorbar={"title":"\u0394\u0394G (kcal/mol)",
                      "titleside":"right", "tickangle":-90}
        ), go.Heatmap(xaxis='x2')]

        layout = go.Layout(
            xaxis=dict( # core positions
                range=[xrange_[0]-0.5, xrange_[-1]+0.5],
                title="G\u03B21 primary structure",
                tickfont=dict(color="#1f77b4"),
                tickmode="array",
                tickangle=-90,
                tickvals=list(xrange_[mask]),
                ticktext=list(wt_seq[mask])
                ),
            xaxis2=dict( # non-core positions
                range=[xrange_[0]-0.5, xrange_[-1]+0.5],
                tickfont=dict(color="#ff7f0e"),
                tickmode="array",
                tickangle=-90,
                tickvals=list(xrange_[~mask]),
                ticktext=list(wt_seq[~mask]),
                overlaying="x"
                ),
            yaxis=dict(
                title="Target",
                tickangle=-90
                ),
            title=title
            )
        fig = go.Figure(data=trace, layout=layout)
        fig.write_html(f"results/heatmap_{ddg_col}_{df.pdbid.iloc[0]}.html")
        fig.show()


def visualize_results(ddgs_per_protein: Dict[str, pd.DataFrame]):
    """
    Allows for visualizing the results:
    - a distribution of the predicted ddGs
    - heatmap of the change of stability landscape, with DSSP info.
    For all flavours of the model (with or without scaling).

    Args:
        preds:  cavity model's predictions, with and without scaling,
                contained in a pd.DataFrame. The predictions column names
                must have 'ddG' in them.
    """

    aa_to_feature_mapper = {
        "D": "acidic", "E": "acidic",
        "H": "basic", "K": "basic", "R": "basic",
        "A": "hydrophobic", "F": "hydrophobic", "I": "hydrophobic",
        "L": "hydrophobic", "M": "hydrophobic", "V": "hydrophobic",
        "Y": "hydrophobic", "W": "hydrophobic",
        "N": "polar", "Q": "polar", "S": "polar", "T": "polar",
        "G": "special", "P": "special", "C": "special",
        }

    preds = ddgs_per_protein.copy() # To avoid inplace modification

    for prot_id in preds:

        # Parse protein variant information.
        preds[prot_id]["resnum"] = preds[prot_id].variant.apply(lambda x: int(x[1:-1]))
        preds[prot_id]["aa"] = preds[prot_id].variant.apply(lambda x: x[0])
        preds[prot_id]["to"] = preds[prot_id].variant.apply(lambda x: x[-1])

        # Classify residues as core, boundary or surface.
        dssp = get_dssp_df_on_file(f"./data/cleaned/{prot_id}.pdb")
        dssp['classification'] = classify_res_by_prot_region(dssp['rel_acc'])

        # Merge dssp info with df variants.
        preds[prot_id]= pd.merge(preds[prot_id], dssp,
                                 how='left', on=['resnum','aa'])

        preds[prot_id]["aa_feature"] = preds[prot_id]["to"].apply(
            lambda x: aa_to_feature_mapper[x])

        # Add Biochemistry category to each residue.
        preds[prot_id]["aa_feature"] = preds[prot_id]["to"].apply(
            lambda x: aa_to_feature_mapper[x])

        # ddg_cols = [col for col in df.columns if "ddg" in col]
        ddg_cols = ["ddg", "ddg_ds_mayo", "ddg_ds_guerois"]
        titles = ["Cavity model",
                    "Cavity + DS (Mayo) models",
                    "Cavity + DS (Guerois) models"]
        heatmap_stability_landscape(preds[prot_id], ddg_cols, titles)

        dataset_distribution(preds[prot_id], ddg_cols, titles)