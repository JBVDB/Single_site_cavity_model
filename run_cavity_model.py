import numpy as np
import pandas as pd
import torch

from collections import OrderedDict
import glob
import os

import Bio.PDB
import Bio.PDB.Polypeptide

from cavity_model import (
    CavityModel,
    ResidueEnvironment,
    ResidueEnvironmentsDataset,
    DownstreamModel,
    ToTensor,
    DDGDataset,
    DDGToTensor,
    DownstreamModel,
)

from helpers import (
    _augment_with_reverse_mutation,
    _populate_dfs_with_nlls_and_nlfs,
    _populate_dfs_with_resenvs,
    _get_ddg_validation_dataloaders,
    _predict_with_downstream,
    _eval_loop
)


def get_resenvs_per_protein(parsed_prots: dict):
    """Get all residue environments per protein."""

    resenvs_per_protein = {}
    for dataset_key, pdbs_wildcard in parsed_prots.items():
        parsed_pdb_filenames = sorted(glob.glob(pdbs_wildcard))

        dataset = ResidueEnvironmentsDataset(parsed_pdb_filenames,
                                             transformer=None)
        dataset_look_up = {}
        for resenv in dataset:
            key = (
                f"{resenv.pdb_id}{resenv.chain_id}_{resenv.pdb_residue_number}"
                f"{Bio.PDB.Polypeptide.index_to_one(resenv.restype_index)}"
            )
            dataset_look_up[key] = resenv
        resenvs_per_protein[dataset_key] = dataset_look_up
    return resenvs_per_protein


def get_variants_per_protein(resenvs_per_protein):

    variants_per_protein = OrderedDict()

    for prot_id in resenvs_per_protein:
        prot_infos = np.load(resenvs_per_protein[prot_id])

        seq_wt_idx = np.argmax(prot_infos["aa_onehot"], axis=1)
        chain_boundary_indices = prot_infos["chain_boundary_indices"]
        res_nbs = prot_infos["residue_numbers"]
        idx_to_one = Bio.PDB.Polypeptide.index_to_one

        chainid, variant = [], []

        for i, (res_nb, restype_idx) in enumerate(zip(res_nbs, seq_wt_idx)):

            for j, boundary in enumerate(chain_boundary_indices):
                if i < boundary:
                    chainid.extend([chr(65 + j-1) for _ in range(19)])
                    break

            variant.extend([f"{idx_to_one(restype_idx)}"
            f"{res_nb}{idx_to_one(x)}" for x in range(20) if x != restype_idx])

        variants_per_protein[prot_id] = pd.DataFrame({
            "pdbid": prot_id.split("_")[0][0:4],
            "chainid": chainid,
            "variant": variant
        })

        return variants_per_protein


def get_raw_predictions(variants_per_protein: dict,
                        device="cuda"):

    cavity_model_infer_net = CavityModel(device).to(device)
    cavity_model_infer_net.load_state_dict(torch.load("models/cavity_model.pt"))
    cavity_model_infer_net.eval()

    _populate_dfs_with_nlls_and_nlfs(
        variants_per_protein, cavity_model_infer_net,
        device, BATCH_SIZE=20, EPS=1e-9
    )

    for prot_id in variants_per_protein.keys():
        variants_per_protein[prot_id].rename({"ddg_pred_no_ds": "ddg"},
                                        axis=1,
                                        inplace=True)
    print("Finished predicting \u0394\u0394Gs with the Cavity Model.")

    return variants_per_protein


def scale_ddg_by_downstream_model(raw_ddgs_per_protein: dict,
                                  mode="guerois",
                                  DEVICE="cuda"):
    # load downstream model
    downstream_model_net = DownstreamModel().to(DEVICE)

    downstream_model_net.load_state_dict(
        torch.load(f"models/ds_model_{mode}.pt"))

    # get validation dataloader
    ddg_dataloaders_val_dict = _get_ddg_validation_dataloaders(
        raw_ddgs_per_protein)

    # get predictions
    ddg_pred = _predict_with_downstream(ddg_dataloaders_val_dict,
                                        downstream_model_net,
                                        DEVICE)

    for prot_id in ddg_pred.keys():
        raw_ddgs_per_protein[prot_id][f"ddg_ds_{mode}"] = ddg_pred[prot_id]
    print(f"Finished scaling the \u0394\u0394Gs with {mode.capitalize()} ssl.")
    return raw_ddgs_per_protein


def run_cavity_model(parsed_prots: list, scaling_mode="all"):
    """
    Predict change of stability values for all possible single-site mutants
    with the cavity model. The predictions are scaled by a downtream model
    based on a site-saturation library of single-site mutants.
    The results are saved in a .csv file.

    Args:
        parsed_prots: 
            list of pdb file paths.
        scaling_mode: default="all"
            which downstream model to use: 'all', 'mayo' or 'guerois'.
    """
    if not isinstance(parsed_prots, list):
        parsed_prots = [parsed_prots]

    parsed_pdbs_wildcards = {}
    for prot_id in parsed_prots:
        parsed_pdbs_wildcards[prot_id] = f"data/parsed/"\
                                         f"{prot_id}_coordinate_features.npz"

    # Create a dict of dict of all residue environments per protein.
    resenvs_per_protein = get_resenvs_per_protein(parsed_pdbs_wildcards)

    # Create a dict of dict of all single-site variants per protein.
    ddgs_per_protein = get_variants_per_protein(parsed_pdbs_wildcards)

    # Combine both datasets' data.
    _populate_dfs_with_resenvs(ddgs_per_protein, resenvs_per_protein)

    # Get raw ddG value predictions from cavity model
    ddgs_per_protein = get_raw_predictions(ddgs_per_protein)

    # Scale the ddGs by a model trained on a specified site-saturation library.
    modes = ["guerois", "mayo"] if scaling_mode == "all" else [scaling_mode]
    for mode in modes:
        ddgs_per_protein = scale_ddg_by_downstream_model(ddgs_per_protein, mode)

    # Save results for each protein.
    for prot_id in ddgs_per_protein:
        ddgs_per_protein[prot_id].to_csv(
            f"results/{prot_id}_{scaling_mode}.csv", index=False)

    return ddgs_per_protein