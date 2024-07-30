import numpy as np
from cosmopower import cosmopower_PCA
from cosmopower import cosmopower_PCAplusNN
import pytest
import os
dirname = os.path.dirname(os.path.abspath(__file__))


def test_train():

    print("Testing cosmopower_PCA and cosmopower_PCAplusNN\n")

    # parameters
    parameters = ["h", "n_s", "ln10^{10}A_s",
                  "omega_cdm", "omega_b", "tau_reio"]

    # modes
    ell_range = np.arange(2, 2509)

    # number of Principal Components
    n_pcas = 64

    # instantiate PCA class
    cp_pca = cosmopower_PCA(parameters=parameters, modes=ell_range,
                            n_pcas=n_pcas,
                            parameters_filenames=[
                                os.path.join(dirname, "tt_data/params_tt_1"),
                                os.path.join(dirname, "tt_data/params_tt_2")
                            ],
                            features_filenames=[
                                os.path.join(dirname,
                                             "tt_data/log_spectra_tt_1"),
                                os.path.join(dirname,
                                             "tt_data/log_spectra_tt_2")
                            ],
                            verbose=True)

    # PCA compression
    cp_pca.transform_and_stack_training_data()
    cp_pca.validate_pca_basis(features_filename=os.path.join(dirname, "tt_data/log_spectra_tt_3"))  # noqa: F401

    # instantiate PCAplusNN class
    cp_pca_nn = cosmopower_PCAplusNN(cp_pca=cp_pca, verbose=True)

    # train model
    cp_pca_nn.train(filename_saved_model=os.path.join(dirname,"PCAplusNN_tt_test"),
                    # cooling schedule
                    validation_split=0.1, learning_rates=[1e-2, 1e-3],
                    batch_sizes=1024, gradient_accumulation_steps=1,
                    # early stopping set up
                    patience_values=5, max_epochs=10)

    # load model
    cp_pca_nn = cosmopower_PCAplusNN(restore_filename=os.path.join(dirname, "PCAplusNN_tt_test"))  # noqa: F401

    # predictions on testing params
    #testing_params = np.load(os.path.join(dirname, "../.tt_data_verysmall/params_tt_3.npz"))  # noqa: F401
    #cp_pca_nn.predictions_np(testing_params)

    # clean up
    os.remove(cp_pca.pca_filename+"_pca.npy")
    os.remove(cp_pca.pca_filename+"_parameters.npy")
    os.remove(os.path.join(dirname,"PCAplusNN_tt_test.npz"))
