# desi-clustering

Collection of scripts to produce the DESI DR2 clustering measurements from the data / mocks catalogs to the parameter inferences.


## 📦 Installation

You can install the latest version directly from the GitHub repository:

```bash
pip install git+https://github.com/cosmodesi/desi-clustering.git
```

Alternatively, if you plan to contribute or modify the code, install in editable (development) mode:

```bash
git clone https://github.com/cosmodesi/desi-clustering.git
cd desi-clustering
pip install -e .
```
In both cases, to compute (fiducial) clustering statistics you can run the command:
```bash
clustering-stats --help
```

## Environment

```
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main  # source the environment
# You may want to have it in the jupyter-kernel for plots
${COSMODESIMODULES}/install_jupyter_kernel.sh main  # this to be done once
```
You may already have the above kernel (corresponding to the standard GQC environment) installed.
In this case, you can delete it:
```
rm -rf $HOME/.local/share/jupyter/kernels/cosmodesi-main
```
and rerun:
```
${COSMODESIMODULES}/install_jupyter_kernel.sh main
```
Note that you may need to restart (close and reopen) your notebooks for the changes to propagate.