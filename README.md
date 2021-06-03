# Single site cavity model: installation

(tested on Colab with miniconda)

0. Clone the repository and change directory

git clone https://github.com/mahermkassem/Cavity_Model_Demo.git
cd Cavity_Model_Demp/

1. Install and activate the provided exported conda environment.

conda create --name cavity-model python=3.6
conda activate cavity-model
conda install pdbfixer=1.5 

2. Install reduce. This program is used by my parser to add missing hydrogens to the proteins

git clone https://github.com/rlabduke/reduce.git
cd reduce/
make; make install # This might give an error but provide the reduce executable in this directory

You should be able to run all the code in the deployment.ipynb notebook.


