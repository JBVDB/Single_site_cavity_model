from pdb_parser_scripts.clean_pdb import clean_pdb
from pdb_parser_scripts.extract_environments import extract_environments

import traceback
import warnings
import os

# !chmod +x reduce/reduce
# !chmod +x pdb_parser_scripts/clean_pdb.py
# !chmod +x pdb_parser_scripts/extract_environments.py


def parse_data(pdb_file,
               clean_pdb_file=True
               ):

    clean_file_path = "."
    pdb_id = os.path.basename(pdb_file).split(".")[0]

    try:
        if clean_pdb_file:
            with warnings.catch_warnings():
                warnings.simplefilter(
                    'ignore', Bio.PDB.PDBExceptions.PDBConstructionWarning
                    )

                clean_file_path = "data/cleaned"
                clean_pdb(pdb_file, clean_file_path, "reduce/reduce")
                pdb_id = f"{pdb_id}_clean"
                print(f"Protein '{pdb_id}' cleaned successfully." )

        extract_environments(f"{clean_file_path}/{pdb_id}.pdb",
                            pdb_id=pdb_id,
                            out_dir="data/parsed")
        print(f"Protein '{pdb_id}' parsed successfully." )
    except Exception:
        error_msg = traceback.format_exc()
        print(f"{pdb_file} failed.\n {error_msg}")