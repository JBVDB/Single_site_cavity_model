# Single site cavity model

3D Convolutional Neural Network predicting the identity probability distribution for each protein residue, based on their direct structure environment.
Beyond the identity prediction, the cavity model estimates the change of stability for any single-site mutation (&#x0394;&#x0394;G). These &#x0394;&#x0394;G values are also scaled non-linearly with a Downstream model based on site saturation libraries *Guerois* [[1]](#1) and *Mayo* [[2]](#2).

This work is based on [[3]](#3). The demo implementation comes from [[4]](#4).

## Models
- **cavity_model.pt** the trained Cavity model, with dataset from [[3]](#3).
- **ds_model_guerois.pt** and **ds_model_mayo.pt** trained Downstream models, respectively with [[1]](#1) and [[2]](#2).

## Code

- **cavity_model.py** contains the main cavity model and downstream model classes and data loaders.
- **helpers.py** protected helper functions for the pipeline.
- **pdb_parser_scripts** contains a pdb cleaning script and a parser script to extract residue environments.
- **visualization.py** for plotting and save predictions results.

For interactive use:
 use with Colaboratory.
- **deployment_jupyter.ipynb** for use with server/local Jupyter notebook.

## How to use
- Insert your raw .pdb files of interest in data/pdbs/raw.
- Use one of two versions of the deployment notebook to get the cavity model's and the two cavity model + Downstream models' predictions.

## Installation

0. Clone the repository and change directory.

```bash
git clone https://github.com/JBVDB/Single_site_cavity_model.git
cd Single_site_cavity_model/
```

1. Install reduce. This program is used by my parser to add missing hydrogens to the proteins.

```bash
git clone https://github.com/rlabduke/reduce.git
cd reduce/
make; make install # This might give an error but provide the reduce executable in this directory
```

### If working with Colaboratory:

2. Simply use the **deployment_colab.ipynb** notebook.

### If working with Shell/Jupyter-notebook (server/local):

2. Get miniconda (or Anaconda) compatible with Python3.6 in current dir.

```bash
MINICONDA_INSTALLER_SCRIPT=Miniconda3-4.5.4-Linux-x86_64.sh
MINICONDA_PREFIX=.
wget -nc https://repo.continuum.io/miniconda/$MINICONDA_INSTALLER_SCRIPT
chmod +x $MINICONDA_INSTALLER_SCRIPT
./$MINICONDA_INSTALLER_SCRIPT -b -f -p $MINICONDA_PREFIX
```

3. Update to most recent compatible version in current dir.

```bash
bash ./Miniconda3-4.5.4-Linux-x86_64.sh
conda install conda=4.9.2  --yes
```

4. Create local environment (default options and writing in .bashrc).

```bash
conda create -n cav_model python=3.6 --yes
```
In case "conda activate cav_model" raises an error:

  - Ini shell for conda:

  ```bash
  conda init bash
  ```

  - Restart shell session and return to Single_site_cavity_model/ dir.

5. Activate env and install dependencies

```bash
conda activate cav_model

conda install -n cav_model -c omnia openmm=7.3.1 pdbfixer=1.5 -y
conda install -n cav_model tqdm pandas=1.1 matplotlib=3.3 seaborn scipy=1.5 plotly>=4.14 -y
conda install -n cav_model -c conda-forge biopython -y
conda install -n cav_model pytorch=1.7 -c pytorch -y
```

**The env is ready for use with shell. If use with a Jupyter-notebook:**

6. Install Jupyter-notebook and create new IPython kernel in env.

```bash
python -m pip install jupyter
python -m ipykernel install --user --name=cav_model
```

7. Launch a jupyter-notebook session

```bash
jupyter-notebook --no-browser  --port 8100 --ip IP_ADDRESS
```
8. Launch **deployment_jupyter.ipynb**. Make sure you use the "cav_model" kernel (Kernel -> Change kernel -> cav_model) and that you activated the env beforehand.

NB: to find your ip:

```bash
hostname -i | cut -f 2 -d " "
```

## References

<a id="1">[1]</a> 
https://doi.org/10.1016/s0022-2836(02)00442-4
   
<a id="2">[2]</a>
https://doi.org/10.1101/484949

<a id="3">[3]</a> 
Boomsma, W & Frellsen, J 2017, Spherical convolutions and their application in molecular modelling. in Advances in Neural Information Processing Systems 30: NIPS 2017. Curran Associates, Inc., pp. 3436-3446. <https://papers.nips.cc/paper/6935-spherical-convolutions-and-their-application-in-molecular-modelling.pdf>

<a id="4">[4]</a> 
https://github.com/mahermkassem/Cavity_Model_Demo
