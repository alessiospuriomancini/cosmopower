import numpy as np
from cosmopower import cosmopower_NN
import pytest
import os
dirname = os.path.dirname(os.path.abspath(__file__))


def test_train():

    print("Testing cosmopower_NN\n")

    # parameters
    parameters = ["h", "n_s", "ln10^{10}A_s",
                  "omega_cdm", "omega_b", "tau_reio"]

    # modes
    ell_range = np.arange(2, 2509)

    # instantiate NN class
    cp_nn = cosmopower_NN(parameters=parameters, modes=ell_range, verbose=True)

    # training parameters
    training_parameters = np.load(os.path.join(dirname, "tt_data/camb_tt_training_params.npz"))  # noqa: F401
    training_features = np.load(os.path.join(dirname, "tt_data/camb_tt_training_log_spectra.npz"))["features"]  # noqa: F401

    # train
    cp_nn.train(training_parameters=training_parameters,
                training_features=training_features,
                filename_saved_model=os.path.join(dirname, "NN_tt_test"),
                # cooling schedule
                validation_split=0.1, learning_rates=[1e-2, 1e-3],
                batch_sizes=1024, gradient_accumulation_steps=1,
                # early stopping set up
                patience_values=5, max_epochs=10)

    # restore model
    cp_nn = cosmopower_NN(restore_filename=os.path.join(dirname,"NN_tt_test"))

    # compute predictions on training params
    #cp_nn.predictions_np(training_parameters)

    # clean up
    os.remove(os.path.join(dirname,"NN_tt_test")+".npz")
