from typing import Dict, List, Union

import numpy as np
import pandas as pd
import torch
import timeit
from Bio.PDB.Polypeptide import index_to_one, one_to_index
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, Dataset
from collections import OrderedDict
from scipy.stats import pearsonr

from cavity_model import (
    CavityModel,
    ResidueEnvironmentsDataset,
    ToTensor,
    DDGDataset,
    DDGToTensor,
    DownstreamModel,
)


def _train_val_split(
    parsed_pdb_filenames: List[str],
    TRAIN_VAL_SPLIT: float,
    DEVICE: str,
    BATCH_SIZE: int,
):
    """
    Helper function to perform training and validation split of ResidueEnvironments. Note that
    we do the split on PDB level not on ResidueEnvironment level due to possible leakage.
    """
    n_train_pdbs = int(len(parsed_pdb_filenames) * TRAIN_VAL_SPLIT)
    filenames_train = parsed_pdb_filenames[:n_train_pdbs]
    filenames_val = parsed_pdb_filenames[n_train_pdbs:]

    to_tensor_transformer = ToTensor(DEVICE)

    dataset_train = ResidueEnvironmentsDataset(
        filenames_train, transformer=to_tensor_transformer # thanks to call function
    )
    dataset_val = ResidueEnvironmentsDataset(
        filenames_val, transformer=to_tensor_transformer
    )

    dataloader_train = DataLoader( # read the data (and shuffle it) within batch size and put into memory.
        dataset_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=to_tensor_transformer.collate_cat, # avoid having to load data to CUDA in the NN model itself!
        drop_last=True, # drop_last=True parameter ignores the last batch (when the number of examples in your dataset is not divisible by your batch_size
    )
    # TODO: Fix it so drop_last doesn't have to be True when calculating validation accuracy.
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=to_tensor_transformer.collate_cat, # if using batch, callable that specifies how the batch is created.
        drop_last=True, # ignores the last batch (when the number of examples in your dataset is not divisible by your batch_size)
    )

    print(
        f"Training data set includes {len(filenames_train)} pdbs with "
        f"{len(dataset_train)} environments."
    )
    print(
        f"Validation data set includes {len(filenames_val)} pdbs with "
        f"{len(dataset_val)} environments."
    )

    return dataloader_train, dataset_train, dataloader_val, dataset_val


def _train_step(
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    cavity_model_net: CavityModel,
    optimizer: torch.optim.Adam,
    loss_function: torch.nn.CrossEntropyLoss,
) -> (torch.Tensor, float):
    """
    Helper function to take a training step
    """
    cavity_model_net.train()
    optimizer.zero_grad()
    batch_y_pred = cavity_model_net(batch_x)
    loss_batch = loss_function(batch_y_pred, torch.argmax(batch_y, dim=-1))
    loss_batch.backward()
    optimizer.step()
    return (batch_y_pred, loss_batch.detach().cpu().item())


def _eval_loop(
    cavity_model_net: CavityModel,
    dataloader_val: DataLoader,
    loss_function: torch.nn.CrossEntropyLoss,
) -> (float, float):
    """
    Helper function to perform an eval loop
    """
    # Eval loop. Due to memory, we don't pass the whole eval set to the model
    labels_true_val = []
    labels_pred_val = []
    loss_batch_list_val = []
    for batch_x_val, batch_y_val in dataloader_val:

        cavity_model_net.eval()
        batch_y_pred_val = cavity_model_net(batch_x_val)

        loss_batch_val = loss_function(
            batch_y_pred_val, torch.argmax(batch_y_val, dim=-1)
        )
        loss_batch_list_val.append(loss_batch_val.detach().cpu().item())

        labels_true_val.append(torch.argmax(batch_y_val, dim=-1).detach().cpu().numpy())
        labels_pred_val.append(
            torch.argmax(batch_y_pred_val, dim=-1).detach().cpu().numpy()
        )
        
    acc_val = np.mean(
        (np.reshape(labels_true_val, -1) == np.reshape(labels_pred_val, -1))
    )
    loss_val = np.mean(loss_batch_list_val)
    return acc_val, loss_val

# in _train_loop, add snipet for saving Train/Val loss and acc and return them.

def _train_loop(
    dataloader_train: DataLoader,
    dataloader_val: DataLoader,
    cavity_model_net: CavityModel,
    loss_function: torch.nn.CrossEntropyLoss,
    optimizer: torch.optim.Adam,
    EPOCHS: int,
    PATIENCE_CUTOFF: int,
    folder="cavity_models_baseline",
    model_name="model"
    
):
    """
    Helper function to perform training loop for the Cavity Model.
    """
    current_best_epoch_idx = -1
    current_best_loss_val = 1e4
    patience = 9 # we start at 0
    epoch_idx_to_model_path = {}
    print("- Starts training...\n")
    for epoch in range(EPOCHS):
        t1 = timeit.default_timer()
        labels_true = []
        labels_pred = []
        loss_batch_list = []
        for batch_x, batch_y in dataloader_train:
            # Take train step
            batch_y_pred, loss_batch = _train_step(
                batch_x, batch_y, cavity_model_net, optimizer, loss_function
            )
            loss_batch_list.append(loss_batch)

            labels_true.append(torch.argmax(batch_y, dim=-1).detach().cpu().numpy())
            labels_pred.append(
                torch.argmax(batch_y_pred, dim=-1).detach().cpu().numpy()
            )
            

        # Train epoch metrics
        acc_train = np.mean(
            (np.reshape(labels_true, -1) == np.reshape(labels_pred, -1))
        )
        loss_train = np.mean(loss_batch_list)

        # Validation epoch metrics
        acc_val, loss_val = _eval_loop(cavity_model_net, dataloader_val, loss_function)

        print(
            f"Epoch {epoch:2d}. Train loss: {loss_train:5.3f}. "
            f"Train Acc: {acc_train:4.2f}. Val loss: {loss_val:5.3f}. "
            f"Val Acc {acc_val:5.3f}"
        )
        
        print(f"Epoch {epoch} done in {round(timeit.default_timer() - t1, 2)} seconds.")
        
        # Save model
        model_path = f"{folder}/{model_name}_epoch_{epoch:02d}.pt"
        epoch_idx_to_model_path[epoch] = model_path
        torch.save(cavity_model_net.state_dict(), model_path)

        # Early stopping
        if loss_val < current_best_loss_val:
            current_best_loss_val = loss_val
            current_best_epoch_idx = epoch
            patience = 0
        else:
            patience += 1
        if patience > PATIENCE_CUTOFF:
            print("Early stopping activated.")
            break
    
    best_model_path = epoch_idx_to_model_path[current_best_epoch_idx]
    print(
        f"Best epoch idx: {current_best_epoch_idx} with validation loss: "
        f"{current_best_loss_val:5.3f}\nFound at: "
        f"'{best_model_path}'"
    )
    return best_model_path
    
    # add a means of saving validation and train errors per epoch in a dict to pickle in order to plot later on.


def _populate_dfs_with_resenvs(
    ddg_data_dict: Dict[str, pd.DataFrame], resenv_datasets_look_up: Dict[str, Dataset]
):
    """
    Helper function populate ddG dfs with the WT ResidueEnvironment objects (in place modification)
    """
    print(
        "Dropping data points where residue is not defined in structure "
        "or due to missing parsed pdb file"
    )
    # Add wt residue environments to standard ddg data dataframes
    for ddg_data_key in ddg_data_dict.keys(): # for each dataset
        resenvs_ddg_data = []
        for idx, row in ddg_data_dict[ddg_data_key].iterrows():
            resenv_key = (
                f"{row['pdbid']}{row['chainid']}_"
                f"{row['variant'][1:-1]}{row['variant'][0]}"
            )
            try:
                if "symmetric" in ddg_data_key:
                    ddg_data_key_adhoc_fix = "symmetric"
                else:
                    ddg_data_key_adhoc_fix = ddg_data_key
                resenv = resenv_datasets_look_up[ddg_data_key_adhoc_fix][resenv_key]
                resenvs_ddg_data.append(resenv)
            except KeyError:
                resenvs_ddg_data.append(np.nan)
        ddg_data_dict[ddg_data_key]["resenv"] = resenvs_ddg_data
        n_datapoints_before = ddg_data_dict[ddg_data_key].shape[0]
        ddg_data_dict[ddg_data_key].dropna(inplace=True)
        n_datapoints_after = ddg_data_dict[ddg_data_key].shape[0]
        print(
            f"dropped {n_datapoints_before - n_datapoints_after:4d} / "
            f"{n_datapoints_before:4d} data points from dataset {ddg_data_key}"
        )

        # Add wt and mt idxs to df (ex: M54A being substitution of Met by Aala at site 54)
        ddg_data_dict[ddg_data_key]["wt_idx"] = ddg_data_dict[ddg_data_key].apply(
            lambda row: one_to_index(row["variant"][0]), axis=1 # saving as index ("A" = 0, "C" = 1, "D" = 2, ...)
        )
        ddg_data_dict[ddg_data_key]["mt_idx"] = ddg_data_dict[ddg_data_key].apply(
            lambda row: one_to_index(row["variant"][-1]), axis=1
        )


def _populate_dfs_with_nlls_and_nlfs(
    ddg_data_dict: Dict[str, pd.DataFrame],
    cavity_model_infer_net: CavityModel,
    DEVICE: str,
    BATCH_SIZE: int,
    EPS: float,
):
    """
    Helper function to populate ddG dfs with predicted negative-log-likelihoods and negative-log-frequencies
    (in_place modification).
    """

    # Load PDB amino acid frequencies used to approximate unfolded states
    pdb_nlfs = -np.log(np.load("data/pdb_frequencies.npz")["frequencies"])

    # Add predicted Nlls and NLFs to ddG dataframes
    for ddg_data_key in ddg_data_dict.keys():
        df = ddg_data_dict[ddg_data_key]

        # Perform predictions on matched residue environments
        ddg_resenvs = list(df["resenv"].values)
        ddg_resenv_dataset = ResidueEnvironmentsDataset(
            ddg_resenvs, transformer=ToTensor(DEVICE)
        )

        # Define dataloader for resenvs matched to ddG data
        ddg_resenv_dataloader = DataLoader(
            ddg_resenv_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False, # Should be set to True!
            collate_fn=ToTensor(DEVICE).collate_cat,
            drop_last=False, #
        )

        # Infer NLLs
        pred_nlls = []
        for batch_x, _ in ddg_resenv_dataloader:
            batch_pred_nlls = (
                -torch.log(softmax(cavity_model_infer_net(batch_x), dim=-1) + EPS)
                .detach()
                .cpu()
                .numpy()
            )
            pred_nlls.append(batch_pred_nlls)
        pred_nlls_list = [row for row in np.vstack(pred_nlls)]

        # Add NLLs to dataframe
        df["nlls"] = pred_nlls_list

        # Isolate WT and MT NLLs and add to dataframe
        df["wt_nll"] = df.apply(lambda row: row["nlls"][row["wt_idx"]], axis=1) # row is 21-long (21 nll)
        df["mt_nll"] = df.apply(lambda row: row["nlls"][row["mt_idx"]], axis=1) # get nll predicted for WT and MT index

        # Add PDB database statistics negative log frequencies to df
        df["wt_nlf"] = df.apply(lambda row: pdb_nlfs[row["wt_idx"]], axis=1)
        df["mt_nlf"] = df.apply(lambda row: pdb_nlfs[row["mt_idx"]], axis=1)

        # Add ddG prediction (without downstream model) # ds being downstream model
        df["ddg_pred_no_ds"] = df.apply(
            lambda row: row["mt_nll"] - row["mt_nlf"] - row["wt_nll"] + row["wt_nlf"],
            axis=1,
        )


def _augment_with_reverse_mutation(ddg_data_dict: Dict[str, pd.DataFrame], rosetta=False):
    """
    Helper function that augments the ddg dfs with the reverse mutations.
    The dict contains deep copies of the original dataframes before copying.
    """

    ddg_data_dict_augmented = OrderedDict()
    for ddg_key in ddg_data_dict:
        ddg_data_df_augmented = (
            ddg_data_dict[ddg_key].copy(deep=True).drop(columns="resenv")
        )
        rows_augmented = []
        for row in ddg_data_df_augmented.iterrows():
            row_cp = row[1].copy(deep=True) # Make a deep copy, including a copy of the data and the indices.
            row_cp.name = str(row_cp.name) + "_augmented"

            # Augment
            row_cp["variant"] = (
                row_cp["variant"][-1] + row_cp["variant"][1:-1] + row_cp["variant"][0]
            )
            row_cp["ddg"] = -1.0 * row_cp["ddg"]
            row_cp["ddg_pred_no_ds"] = -1.0 * row_cp["ddg_pred_no_ds"]
            if rosetta:
                row_cp["ddg_rosetta"] = -1.0 * row_cp["ddg_rosetta"]

            wt_idx, mt_idx, wt_nll, mt_nll, wt_nlf, mt_nlf = (
                row_cp["mt_idx"],
                row_cp["wt_idx"],
                row_cp["mt_nll"],
                row_cp["wt_nll"],
                row_cp["mt_nlf"],
                row_cp["wt_nlf"],
            )
            (
                row_cp["wt_idx"],
                row_cp["mt_idx"],
                row_cp["wt_nll"],
                row_cp["mt_nll"],
                row_cp["wt_nlf"],
                row_cp["mt_nlf"],
            ) = (wt_idx, mt_idx, wt_nll, mt_nll, wt_nlf, mt_nlf)

            rows_augmented.append(row_cp)
        ddg_data_df_augmented = ddg_data_df_augmented.append(rows_augmented)
        ddg_data_dict_augmented[ddg_key] = ddg_data_df_augmented

    return ddg_data_dict_augmented


def _get_ddg_training_dataloaders(ddg_data_dict_augmented,
    BATCH_SIZE_DDG,
    SHUFFLE_DDG,
    ):
    """
    Get dataloaders of ddG for Downstream model.
    """
    ddg_dataloaders_train_dict = {}
    for key in ddg_data_dict_augmented.keys(): # get the input shape as we need it before creating the dataloader (see ipynb)
        ddg_dataset_aug = DDGDataset(
            ddg_data_dict_augmented[key], transformer=DDGToTensor()
        )
        ddg_dataloader_aug = DataLoader(
            ddg_dataset_aug,
            batch_size=BATCH_SIZE_DDG,
            shuffle=SHUFFLE_DDG,
            drop_last=True,
        )
        ddg_dataloaders_train_dict[key] = ddg_dataloader_aug

    return ddg_dataloaders_train_dict


def _get_ddg_validation_dataloaders(
    ddg_data_dict,
):
    """
    Helper function that return validation set dataloaders for ddg data.
    """
    keys = ddg_data_dict.keys()
    ddg_dataloaders_val_dict = {}
    for key in keys:
        ddg_dataset = DDGDataset(ddg_data_dict[key], transformer=DDGToTensor())
        ddg_dataloader = DataLoader(
            ddg_dataset,
            batch_size=len(ddg_dataset),  # Full dataset as batch size
            shuffle=False,
            drop_last=False,
        )
        ddg_dataloaders_val_dict[key] = ddg_dataloader

    return ddg_dataloaders_val_dict


def _train_downstream_and_evaluate(
    ddg_dataloaders_train_dict: DataLoader,
    ddg_dataloaders_val_dict: DataLoader,
    downstream_model: DownstreamModel,
    DEVICE,
    LEARNING_RATE_DDG,
    EPOCHS_DDG,
    folder="cavity_models_baseline",
    model_name="ds_model"
):
    pearsons_r_results_dict = {}
    print("start training...\n")
    for train_key in ["guerois"]: #["dms", "protein_g", "guerois"]: # we train the model one dataset at a time.
        print(f"- training on experimental dataset '{train_key}'")
        # Define model
        downstream_model_net = downstream_model().to(DEVICE)
        loss_ddg = torch.nn.MSELoss()
        optimizer_ddg = torch.optim.Adam(
            downstream_model_net.parameters(), lr=LEARNING_RATE_DDG
        )

        t1 = timeit.default_timer()
        for epoch in range(EPOCHS_DDG):
            for batch in ddg_dataloaders_train_dict[train_key]:
                x_batch, y_batch = batch["x_"].to(DEVICE), batch["y_"].float().to(DEVICE)

                downstream_model_net.train()
                optimizer_ddg.zero_grad()
                y_batch_pred = downstream_model_net(x_batch).squeeze()
                loss_ddg_batch = loss_ddg(y_batch_pred, y_batch)
                loss_ddg_batch.backward()
                optimizer_ddg.step()
            
            previous_key = ""
            for val_key in ["dms", "protein_g", "guerois"]: # validation on one batch (which is len(dataset))
                if val_key == train_key:
                    val_key = previous_key
                    continue
                val_batch = next(iter(ddg_dataloaders_val_dict[val_key]))
                val_batch_x, val_batch_y = (
                    val_batch["x_"].to(DEVICE),
                    val_batch["y_"].float(),
                )
                val_batch_y_pred = (
                    downstream_model_net(val_batch_x).reshape(-1).detach().cpu().numpy()
                )
                pearson_r = pearsonr(val_batch_y_pred, val_batch_y)[0]
                if train_key not in pearsons_r_results_dict:
                    pearsons_r_results_dict[train_key] = {}
                if val_key not in pearsons_r_results_dict[train_key]:
                    pearsons_r_results_dict[train_key][val_key] = []

                pearsons_r_results_dict[train_key][val_key].append(pearson_r)
                previous_key = val_key
            if (epoch+1) % 10 == 0:
                print(
                    f"{epoch+1} / {EPOCHS_DDG}, pearson correlation: {pearson_r: .4f} ('{val_key}' dataset, "
                    f"{round((timeit.default_timer() - t1)/10, 2)} seconds/epoch."
                )
                t1 = timeit.default_timer()
        # Save the 3 models
        torch.save(downstream_model_net.state_dict(), f"{folder}/{model_name}_trained_with_{train_key}.pt")
        print(f"model saved at: '{folder}/{model_name}_trained_with_{train_key}.pt'.")
    return pearsons_r_results_dict


def _predict_with_downstream(
    ddg_dataloaders_val_dict,
    downstream_model_net: DownstreamModel,
    DEVICE):
    keys = ddg_dataloaders_val_dict.keys()
    floater = np.vectorize(lambda x: float(x))
    ddg_pred = {}
    downstream_model_net = downstream_model_net.to(DEVICE)
    downstream_model_net.eval()
    for dms_family in keys: # ["dms", "protein_g", "guerois"]:
        ddg_pred[dms_family] = []
        with torch.no_grad():
            # predictions per batch
            for batch in ddg_dataloaders_val_dict[dms_family]:
                x_batch, y_batch = batch["x_"].to(DEVICE), batch["y_"].float().to(DEVICE)
                y_batch_pred = list(downstream_model_net(x_batch).squeeze().to("cpu"))
                
                # save predictions
                try:
                    # ddg_pred[dms_family].extend(list(y_batch_pred.detach().to("cpu").float()))
                    ddg_pred[dms_family].extend(floater(y_batch_pred))
                    
                except Exception as e:
                    print(e.message, e.args)
            print(f"Done validating {dms_family} dataset.")
    return ddg_pred
            
       