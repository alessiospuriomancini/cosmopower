#!/usr/bin/env python
# Author: Alessio Spurio Mancini

import numpy as np
from tqdm import trange
from typing import Sequence
from sklearn.decomposition import IncrementalPCA


# =================================
#               PCA
# =================================
class cosmopower_PCA():
    r"""
    Principal Component Analysis of (log)-power spectra

    Attributes:
        parameters (list):
            model parameters, sorted in the desired order
        modes (numpy.ndarray):
            multipoles or k-values in the (log)-spectra
        n_pcas (int):
            number of PCA components
        verbose (bool):
            whether to print messages at intermediate steps or not
    """

    def __init__(self, parameters: Sequence[str], modes: np.ndarray,
                 n_pcas: int, n_batches: int, verbose: bool = False) -> None:
        r"""
        Constructor
        """
        # attributes
        self.parameters = parameters
        self.n_parameters = len(parameters)
        self.modes = modes
        self.n_modes = len(self.modes)
        self.n_pcas = n_pcas
        self.n_batches = n_batches

        # PCA object
        self.PCA = IncrementalPCA(n_components=self.n_pcas)
        self.trained = False

        # verbose
        self.verbose = verbose

        # print initialization info, if verbose
        if self.verbose:
            print(f"\nInitialized cosmopower_PCA compression with \
                    {self.n_pcas} components \n")

    def dict_to_ordered_arr_np(self, input_dict: dict) -> np.ndarray:
        r"""
        Sort input parameters

        Parameters:
            input_dict (dict [numpy.ndarray]):
                input dict of (arrays of) parameters to be sorted

        Returns:
            numpy.ndarray:
                input parameters sorted according to `parameters`
        """
        if self.parameters is not None:
            return np.stack([input_dict[k] for k in self.parameters], axis=1)
        else:
            return np.stack([input_dict[k] for k in input_dict], axis=1)

    def standardise_features_and_parameters(self, features: np.ndarray,
                                            parameters: np.ndarray
                                            ) -> None:
        r"""
        Compute mean and std for (log)-spectra and parameters
        """
        # mean and std
        self.features_mean = np.zeros(self.n_modes)
        self.features_std = np.zeros(self.n_modes)
        self.parameters_mean = np.zeros(self.n_parameters)
        self.parameters_std = np.zeros(self.n_parameters)

        # loop over training data files, accumulate means and stds
        for i in range(self.n_batches):
            # accumulate
            self.features_mean += np.mean(features, axis=0) / self.n_batches
            self.features_std += np.std(features, axis=0) / self.n_batches
            self.parameters_mean += (np.mean(parameters, axis=0)
                                     / self.n_batches)
            self.parameters_std += np.std(parameters, axis=0) / self.n_batches

    def train_pca(self, features: np.ndarray) -> None:
        r"""
        Train PCA incrementally
        """
        # loop over training data files, increment PCA
        with trange(self.n_batches) as t:
            for i in t:
                # load (log)-spectra and mean+std
                normalised_features = (
                    features - self.features_mean
                ) / self.features_std

                # partial PCA fit
                self.PCA.partial_fit(normalised_features)

    def transform_and_stack_training_data(self, training_features: np.ndarray,
                                          training_parameters: np.ndarray,
                                          filename: str = './tmp',
                                          retain: bool = True) -> None:
        r"""
        Transform the training data set to PCA basis

        Parameters:
            filename (str):
                filename tag (no suffix) for PCA coefficients and parameters
            retain (bool):
                whether to retain training data as attributes
        """
        if self.verbose:
            print("starting PCA compression")
        self.standardise_features_and_parameters()
        self.train_pca()

        # transform the (log)-spectra to PCA basis
        training_pca = np.concatenate([
            self.PCA.transform(
                (training_features - self.features_mean) / self.features_std
            ) for i in range(self.n_batches)
        ])

        # mean and std of PCA basis
        self.pca_mean = np.mean(training_pca, axis=0)
        self.pca_std = np.std(training_pca, axis=0)

        # save stacked transformed training data
        self.pca_filename = filename
        np.save(self.pca_filename + '_pca.npy', training_pca)
        np.save(self.pca_filename + '_parameters.npy', training_parameters)

        # retain training data as attributes if retain == True
        if retain:
            self.training_pca = training_pca
            self.training_parameters = training_parameters
        if self.verbose:
            print("PCA compression done")
            if retain:
                print("parameters and PCA coefficients of training set \
                       stored in memory")

    def validate_pca_basis(self, features_filename: str
                           ) -> tuple[np.ndarray, np.ndarray]:
        r"""
        Validate PCA given some validation data

        Parameters:
            features_filename (str):
                filename tag (no suffix) for validation (log)-spectra

        Returns:
            features_pca (numpy.ndarray):
                PCA of validation (log)-spectra
            features_in_basis (numpy.ndarray):
                inverse PCA transform of validation (log)-spectra
        """
        # load (log)-spectra and standardise
        features = np.load(features_filename + ".npz")["features"]
        normalised_features = (
            features - self.features_mean
        ) / self.features_std

        # transform to PCA basis and back
        features_pca = self.PCA.transform(normalised_features)
        features_in_basis = np.dot(features_pca,
                                   self.pca_transform_matrix) * \
                            self.features_std + self.features_mean  # noqa E127

        # return PCA coefficients and (log)-spectra in basis
        return features_pca, features_in_basis

    @property
    def pca_transform_matrix(self) -> np.ndarray:
        return self.PCA.components_

    @property
    def is_compressed(self) -> bool:
        return self.trained
