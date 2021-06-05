# Single site cavity model: installation

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

## If working with Colaboratory:

2. Simply use the **deployment_colab.ipynb** notebook.

## If working with Shell/Jupyter-notebook (server/local):

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
conda install -n cav_model tqdm pandas>=1.1 -y
conda install -n cav_model -c conda-forge biopython -y
conda install -n cav_model pytorch=1.7 torchvision -c pytorch -y
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
8. Launch **deployment_jupyter.ipynb**.

Make sure you use the "cav_model" kernel (Kernel -> Change kernel -> cav_model)

NB: to find your ip:

```bash
hostname -i | cut -f 2 -d " "

