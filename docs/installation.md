# Installation

We recommend installing ``CosmoPower`` within a [Conda](https://docs.conda.io/projects/conda/en/latest/index.html) virtual environment. 
For example, to create and activate an environment called ``cp_env``, use:

    conda create -n cp_env python=3.7 pip && conda activate cp_env

Once inside the environment, the user can choose to install ``CosmoPower``:

- **from PyPI**

        pip install cosmopower

- **from source**

        git clone https://github.com/alessiospuriomancini/cosmopower_public
        cd cosmopower
        pip install -e .

    To test the installation you can use

        pytest

