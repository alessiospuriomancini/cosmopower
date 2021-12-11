# Installation

We recommend installing ``CosmoPower`` within a [Conda](https://docs.conda.io/projects/conda/en/latest/index.html) virtual environment. 
For example, to create and activate an environment called ``cp_env``, use:

    conda create -n cp_env python=3.7 pip && conda activate cp_env

Once inside the environment, you can install ``CosmoPower``:

- **from PyPI**

        pip install cosmopower

    To test the installation, you can use

        python3 -c 'import cosmopower as cp'
    
    If you do not have a GPU on your machine, you will see a warning message about it which you can safely ignore.

- **from source**

        git clone https://github.com/alessiospuriomancini/cosmopower
        cd cosmopower
        pip install .

    To test the installation, you can use

        pytest
