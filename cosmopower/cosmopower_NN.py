#!/usr/bin/env python
# Author: Alessio Spurio Mancini

import os
import warnings
import numpy as np
import tensorflow as tf
from typing import Sequence
from tqdm import trange
dtype = tf.float32


# =================================
#               NN
# =================================
class cosmopower_NN(tf.keras.Model):
    r"""
    Mapping between cosmological parameters and (log)-power spectra

    Attributes:
        parameters (list [str]):
            model parameters, sorted in the desired order
        modes (numpy.ndarray):
            multipoles or k-values in the (log)-spectra
        parameters_mean (numpy.ndarray):
            mean of input parameters
        parameters_std (numpy.ndarray):
            std of input parameters
        features_mean (numpy.ndarray):
            mean of output features
        features_std (numpy.ndarray):
            std of output features
        n_hidden (list [int]):
            number of nodes for each hidden layer
        restore (bool):
            whether to restore a previously trained model or not
        restore_filename (str):
            filename tag (without suffix) for restoring trained model from file
            (this will be a pickle file with all of the model attributes and weights)
        trainable (bool):
            training layers
        optimizer (tf.keras.optimizers):
            optimizer for training
        verbose (bool):
            whether to print messages at intermediate steps or not
        allow_pickle (bool):
            whether to permit the (legacy) loading of .pkl files.
    """

    def __init__(self, parameters: Optional[Sequence[str]] = None,
                 modes: Optional[np.ndarray] = None,
                 parameters_mean: Optional[np.ndarray] = None,
                 parameters_std: Optional[np.ndarray] = None,
                 features_mean: Optional[np.ndarray] = None,
                 features_std: Optional[np.ndarray] = None,
                 n_hidden: Sequence[int] = [512, 512, 512],
                 restore_filename: Optional[str] = None,
                 trainable: bool = True,
                 optimizer: Optional[tf.keras.Optimizer] = None,
                 verbose: bool = False, allow_pickle: bool = False) -> None:
        """
        Constructor
        """
        # super
        super(cosmopower_NN, self).__init__()

        # restore
        if restore_filename is not None:
            self.restore(restore_filename, allow_pickle=allow_pickle)

        # else set variables from input arguments
        else:
            # attributes
            self.parameters = parameters
            self.n_parameters = len(self.parameters)
            self.modes = modes
            self.n_modes = len(self.modes)
            self.n_hidden = n_hidden

            # architecture
            self.architecture = [self.n_parameters] + self.n_hidden + [self.n_modes]
            self.n_layers = len(self.architecture) - 1

            # input parameters mean and std
            self.parameters_mean_ = parameters_mean if parameters_mean is not None else np.zeros(self.n_parameters)
            self.parameters_std_ = parameters_std if parameters_std is not None else np.ones(self.n_parameters)

            # (log)-spectra mean and std
            self.features_mean_ = features_mean if features_mean is not None else np.zeros(self.n_modes)
            self.features_std_ = features_std if features_std is not None else np.ones(self.n_modes)

        # input parameters mean and std
        self.parameters_mean = tf.constant(self.parameters_mean_, dtype=dtype, name='parameters_mean')
        self.parameters_std = tf.constant(self.parameters_std_, dtype=dtype, name='parameters_std')

        # (log)-spectra mean and std
        self.features_mean = tf.constant(self.features_mean_, dtype=dtype, name='features_mean')
        self.features_std = tf.constant(self.features_std_, dtype=dtype, name='features_std')

        # weights, biases and activation function parameters for each layer of the network
        self.W = []
        self.b = []
        self.alphas = []
        self.betas = []
        for i in range(self.n_layers):
            self.W.append(tf.Variable(tf.random.normal([self.architecture[i], self.architecture[i+1]], 0., 1e-3), name="W_" + str(i), trainable=trainable))
            self.b.append(tf.Variable(tf.zeros([self.architecture[i+1]]), name="b_" + str(i), trainable=trainable))

        for i in range(self.n_layers-1):
            self.alphas.append(tf.Variable(tf.random.normal([self.architecture[i+1]]), name="alphas_" + str(i), trainable=trainable))
            self.betas.append(tf.Variable(tf.random.normal([self.architecture[i+1]]), name="betas_" + str(i), trainable=trainable))

        # restore weights if restoring
        if restore_filename is not None:
            for i in range(self.n_layers):
                self.W[i].assign(self.W_[i])
                self.b[i].assign(self.b_[i])
            for i in range(self.n_layers-1):
                self.alphas[i].assign(self.alphas_[i])
                self.betas[i].assign(self.betas_[i])

        # optimizer
        self.optimizer = optimizer or tf.keras.optimizers.Adam()
        self.verbose= verbose

        # print initialization info, if verbose
        if self.verbose:
            multiline_str = "\nInitialized cosmopower_NN model, \n" \
                            f"mapping {self.n_parameters} input parameters to {self.n_modes} output modes, \n" \
                            f"using {len(self.n_hidden)} hidden layers, \n" \
                            f"with {list(self.n_hidden)} nodes, respectively."
            print(multiline_str)

    def activation(self, x: tf.Tensor, alpha: tf.Tensor, beta: tf.Tensor
                   ) -> tf.Tensor:
        r"""
        Non-linear activation function

        Parameters:
            x (Tensor):
                linear output from previous layer
            alpha (Tensor):
                trainable parameter
            beta (Tensor):
                trainable parameter

        Returns:
            Tensor:
                the result of applying the non-linear activation function to the linear output of the layer
        """
        return tf.multiply(tf.add(beta,
                                  tf.multiply(tf.sigmoid(tf.multiply(alpha,
                                                                     x)),
                                              tf.subtract(1.0, beta))), x)

    @tf.function
    def predictions_tf(self,
                       parameters_tensor):
        r"""
        Prediction given tensor of input parameters,
        fully implemented in TensorFlow

        Parameters:
            parameters_tensor (Tensor):
                input parameters
            training (bool):
                whether or not we are currently training

        Returns:
            Tensor:
                output predictions
        """
        outputs = []
        layers = [tf.divide(tf.subtract(parameters_tensor, self.parameters_mean), self.parameters_std)]
        for i in range(self.n_layers - 1):
            # linear network operation
            outputs.append(tf.add(tf.matmul(layers[-1], self.W[i]), self.b[i]))
            # non-linear activation function
            layers.append(self.activation(outputs[-1], self.alphas[i], self.betas[i]))

        # linear output layer
        layers.append(tf.add(tf.matmul(layers[-1], self.W[-1]), self.b[-1]))
        # rescale -> output predictions
        return tf.add(tf.multiply(layers[-1], self.features_std), self.features_mean)


    # tensor 10.**predictions
    @tf.function
    def ten_to_predictions_tf(self,
                           parameters_tensor):
        r"""
        10^predictions given tensor of input parameters,
        fully implemented in TensorFlow. It raises 10 to the output
        of ``predictions_tf``

        Parameters:
            parameters_tensor (Tensor):
                input parameters

        Returns:
            Tensor:
                10^output predictions
        """
        return tf.pow(10., self.predictions_tf(parameters_tensor))


# ============= SAVE/LOAD model =============

    # save network parameters to Numpy arrays
    def update_emulator_parameters(self):
        r"""
        Update emulator parameters before saving them
        """
        # put network parameters to numpy arrays
        self.W_ = [self.W[i].numpy() for i in range(self.n_layers)]
        self.b_ = [self.b[i].numpy() for i in range(self.n_layers)]
        self.alphas_ = [self.alphas[i].numpy() for i in range(self.n_layers-1)]
        self.betas_ = [self.betas[i].numpy() for i in range(self.n_layers-1)]

        # put mean and std parameters to numpy arrays
        self.parameters_mean_ = self.parameters_mean.numpy()
        self.parameters_std_ = self.parameters_std.numpy()
        self.features_mean_ = self.features_mean.numpy()
        self.features_std_ = self.features_std.numpy()


    def save(self, filename: str) -> None:
        r"""
        Save network parameters

        Parameters:
            filename (str):
                filename tag (without suffix) where model will be saved
        """
        # Save data as compressed numpy file.
        attributes = { }
        attributes["architecture"] = self.architecture
        attributes["n_layers"] = self.n_layers
        attributes["n_hidden"] = self.n_hidden
        attributes["n_parameters"] = self.n_parameters
        attributes["n_modes"] = self.n_modes

        attributes["parameters"] = self.parameters
        attributes["modes"] = self.modes

        attributes["parameters_mean"] = self.parameters_mean.numpy()
        attributes["parameters_std"] = self.parameters_std.numpy()
        attributes["features_mean"] = self.features_mean.numpy()
        attributes["features_std"] = self.features_std.numpy()

        for i in range(self.n_layers):
            attributes[f"W_{i}"] = self.W[i].numpy()
            attributes[f"b_{i}"] = self.b[i].numpy()
        for i in range(self.n_layers-1):
            attributes[f"alphas_{i}"] = self.alphas[i].numpy()
            attributes[f"betas_{i}"] = self.betas[i].numpy()

        with open(filename + ".npz", "wb") as fp:
            np.savez_compressed(fp, **attributes)


    def restore(self, filename: str, allow_pickle: bool = False) -> None:
        r"""
        Load pre-trained model.
        The default file format is compressed numpy files (.npz). The
        Module will attempt to use this as a file extension and restore
        from there (i.e. look for `filename.npz`). If this file does
        not exist, and `allow_pickle` is set to True, then the file
        `filename.pkl` will be attempted to be read by `restore_pickle`.

        The function will trim the file extension from `filename`, so
        `restore("filename")` and `restore("filename.npz")` are identical.

        Parameters:
        :param filename: filename (without suffix) where model was saved.
        :param allow_pickle: whether or not to permit passing this filename to the `restore_pickle` function.
        """
        # Check if npz file exists.
        filename_npz = filename + ".npz"
        if not os.path.exists(filename_npz):
            # Can we load this file as a pickle file?
            filename_pkl = filename + ".pkl"
            if allow_pickle and os.path.exists(filename_pkl):
                self.restore_pickle(filename_pkl)
                return

            raise IOError(f"Failed to restore network from {filename}: " +
                (" is a pickle file, try setting 'allow_pickle = True'" if os.path.exists(filename_pkl) else " does not exist."))

        with open(filename_npz, "rb") as fp:
            fpz = np.load(fp)

            self.architecture = fpz["architecture"]
            self.n_layers = fpz["n_layers"]
            self.n_hidden = fpz["n_hidden"]
            self.n_parameters = fpz["n_parameters"]
            self.n_modes = fpz["n_modes"]

            self.parameters = list(fpz["parameters"])
            self.modes = fpz["modes"]

            self.parameters_mean_ = fpz["parameters_mean"]
            self.parameters_std_ = fpz["parameters_std"]
            self.features_mean_ = fpz["features_mean"]
            self.features_std_ = fpz["features_std"]

            self.W_ = [ fpz[f"W_{i}"] for i in range(self.n_layers) ]
            self.b_ = [ fpz[f"b_{i}"] for i in range(self.n_layers) ]
            self.alphas_ = [ fpz[f"alphas_{i}"] for i in range(self.n_layers-1) ]
            self.betas_ = [ fpz[f"betas_{i}"] for i in range(self.n_layers-1) ]

    def restore_pickle(self, filename: str) -> None:
        r"""
        Legacy function for restoring model from pickle (.pkl) file.

        This function might be deprecated in the future, due to the way pickle files are read.

        Parameters:
        :param filename: filename (with suffix) where model was saved.
        """
        warnings.warn("CosmoPower pickle files might be deprecated at some point in the future. It is recommended that you save your networks as npz files.", DeprecationWarning)

        if not os.path.exists(filename):
            raise IOError(f"Failed to restore network from {filename}: does not exist.")

        import pickle
        with open(filename, "rb") as fp:
            W, b, alphas, betas, parameters_mean, parameters_std, features_mean, features_std, n_parameters, parameters, n_modes, modes, n_hidden, n_layers, architecture = pickle.load(fp)

        self.architecture = architecture
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_parameters = n_parameters
        self.n_modes = n_modes

        self.parameters = parameters
        self.modes = modes

        self.parameters_mean_ = parameters_mean
        self.parameters_std_ = parameters_std
        self.features_mean_ = features_mean
        self.features_std_ = features_std

        self.W_ = [ W[i] for i in range(self.n_layers) ]
        self.b_ = [ b[i] for i in range(self.n_layers) ]
        self.alphas_ = [ alphas[i] for i in range(self.n_layers-1) ]
        self.betas_ = [ betas[i] for i in range(self.n_layers-1) ]


# ========== NUMPY implementation ===============

    # auxiliary function to sort input parameters
    def dict_to_ordered_arr_np(self,
                               input_dict,
                               ):
        r"""
        Sort input parameters

        Parameters:
            input_dict (dict [numpy.ndarray]):
                input dict of (arrays of) parameters to be sorted

        Returns:
            numpy.ndarray:
                parameters sorted according to desired order
        """
        input_dict = { k : np.atleast_1d(input_dict[k]) for k in input_dict }

        if self.parameters is not None:
            return np.stack([input_dict[k] for k in self.parameters], axis=1)
        else:
            return np.stack([input_dict[k] for k in input_dict], axis=1)


    # forward prediction given input parameters implemented in Numpy
    def forward_pass_np(self,
                        parameters_arr):
        r"""
        Forward pass through the network to predict the output,
        fully implemented in Numpy

        Parameters:
            parameters_arr (numpy.ndarray):
                array of input parameters

        Returns:
          numpy.ndarray:
            output predictions
        """
        # forward pass through the network
        act = []
        layers = [(parameters_arr - self.parameters_mean_)/self.parameters_std_]
        for i in range(self.n_layers-1):

            # linear network operation
            act.append(np.dot(layers[-1], self.W_[i]) + self.b_[i])

            # pass through activation function
            layers.append((self.betas_[i] + (1.-self.betas_[i])*1./(1.+np.exp(-self.alphas_[i]*act[-1])))*act[-1])

        # final (linear) layer -> (standardised) predictions
        layers.append(np.dot(layers[-1], self.W_[-1]) + self.b_[-1])

        # rescale and output
        return layers[-1]*self.features_std_ + self.features_mean_


    # Numpy array predictions
    def predictions_np(self,
                       parameters_dict,
                       ):
        r"""
        Predictions given input parameters collected in a dict.
        Fully implemented in Numpy. Calls ``forward_pass_np``
        after ordering the input parameter dict

        Parameters:
            parameters_dict (dict [numpy.ndarray]):
                dictionary of (arrays of) parameters

        Returns:
            numpy.ndarray:
                output predictions
        """
        parameters_arr = self.dict_to_ordered_arr_np(parameters_dict)
        return self.forward_pass_np(parameters_arr)


    # Numpy array 10.**predictions
    def ten_to_predictions_np(self,
                            parameters_dict,
                            ):
        r"""
        10^predictions given input parameters collected in a dict.
        Fully implemented in Numpy. It raises 10 to the output
        from ``forward_pass_np``

        Parameters:
            parameters_dict (dict [numpy.ndarray]):
                dictionary of (arrays of) parameters

        Returns:
            numpy.ndarray:
                10^output predictions
        """
        return 10.**self.predictions_np(parameters_dict)



    ### Infrastructure for network training ###

    @tf.function
    def compute_loss(self,
                     training_parameters,
                     training_features):
        r"""
        Mean squared difference

        Parameters:
            training_parameters (Tensor):
                input parameters
            training_features (Tensor):
                true features

        Returns:
            Tensor:
                mean squared difference
        """
        return tf.sqrt(tf.reduce_mean(tf.math.squared_difference(self.predictions_tf(training_parameters), training_features)))


    @tf.function
    def compute_loss_and_gradients(self,
                                   training_parameters,
                                   training_features):
        r"""
        Computes mean squared difference and gradients

        Parameters:
            training_parameters (Tensor):
                input parameters
            training_features (Tensor):
                true features
            train (bool):
                whether we are training or not

        Returns:
            loss (Tensor):
                mean squared difference
            gradients (Tensor):
                gradients
        """
        # compute loss on the tape
        with tf.GradientTape() as tape:

            # loss
            loss = tf.sqrt(tf.reduce_mean(tf.math.squared_difference(self.predictions_tf(training_parameters), training_features)))

        # compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)

        return loss, gradients


    def training_step(self,
                      training_parameters,
                      training_features
                      ):
        r"""
        Minimize loss

        Parameters:
            training_parameters (Tensor):
                input parameters
            training_features (Tensor):
                true features

        Returns:
            loss (Tensor):
                mean squared difference
        """
        # compute loss and gradients
        loss, gradients = self.compute_loss_and_gradients(training_parameters, training_features)

        # apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss


    def training_step_with_accumulated_gradients(self,
                                                 training_parameters,
                                                 training_features,
                                                 accumulation_steps=10):
        r"""
        Minimize loss, breaking calculation into accumulated gradients

        Parameters:
            training_parameters (Tensor):
                input parameters
            training_features (Tensor):
                true features
            accumulation_steps (int):
                number of accumulated gradients

        Returns:
            accumulated_loss (Tensor):
                mean squared difference
        """
        # create dataset to do sub-calculations over
        dataset = tf.data.Dataset.from_tensor_slices((training_parameters, training_features)).batch(int(training_features.shape[0]/accumulation_steps))

        # initialize gradients and loss (to zero)
        accumulated_gradients = [tf.Variable(tf.zeros_like(variable), trainable=False) for variable in self.trainable_variables]
        accumulated_loss = tf.Variable(0., trainable=False)

        # loop over sub-batches
        for training_parameters_, training_features_, in dataset:

            # calculate loss and gradients
            loss, gradients = self.compute_loss_and_gradients(training_parameters_, training_features_)

            # update the accumulated gradients and loss
            for i in range(len(accumulated_gradients)):
                accumulated_gradients[i].assign_add(gradients[i]*training_features_.shape[0]/training_features.shape[0])
            accumulated_loss.assign_add(loss*training_features_.shape[0]/training_features.shape[0])

            # apply accumulated gradients
            self.optimizer.apply_gradients(zip(accumulated_gradients, self.trainable_variables))

        return accumulated_loss


# ==========================================
#         main TRAINING function
# ==========================================
    def train(self,
              training_data,
              filename_saved_model,
              # cooling schedule
              validation_split=0.1,
              learning_rates=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
              batch_sizes=1000,
              gradient_accumulation_steps = 1,
              # early stopping set up
              patience_values = 100,
              max_epochs = 1000,
             ):
        r"""
        Train the model

        Parameters:
            training_data (list[Dataset]):
                list of Datasets that contain training data.
            filename_saved_model (str):
                filename tag where model will be saved
            validation_split (float):
                percentage of training data used for validation
            learning_rates (list [float]):
                learning rates for each step of learning schedule
            batch_sizes (list [int]):
                batch sizes for each step of learning schedule
            gradient_accumulation_steps (list [int]):
                batches for gradient accumulations for each step of learning schedule
            patience_values (list [int]):
                early stopping patience for each step of learning schedule
            max_epochs (list [int]):
                maximum number of epochs for each step of learning schedule
        """
        n_iter = len(learning_rates)
        if type(batch_sizes) != list: batch_sizes = n_iter * [batch_sizes]
        if type(gradient_accumulation_steps) != list: gradient_accumulation_steps = n_iter * [gradient_accumulation_steps]
        if type(patience_values) != list: patience_values = n_iter * [patience_values]
        if type(max_epochs) != list: max_epochs = n_iter * [max_epochs]

        # check correct number of steps
        assert len(learning_rates)==len(batch_sizes)\
               ==len(gradient_accumulation_steps)==len(patience_values)==len(max_epochs), \
               'Number of learning rates, batch sizes, gradient accumulation steps, patience values and max epochs are not matching!'

        # training start info, if verbose
        if self.verbose:
            multiline_str = "Starting cosmopower_NN training, \n" \
                            f"using {int(100*validation_split)} per cent of training samples for validation. \n" \
                            f"Performing {len(learning_rates)} learning steps, with \n" \
                            f"{list(learning_rates)} learning rates \n" \
                            f"{list(batch_sizes)} batch sizes \n" \
                            f"{list(gradient_accumulation_steps)} gradient accumulation steps \n" \
                            f"{list(patience_values)} patience values \n" \
                            f"{list(max_epochs)} max epochs \n"
            print(multiline_str)

        #training_parameters = self.dict_to_ordered_arr_np(training_parameters)
        training_parameters = None
        training_features = None

        progress_file = open(filename_saved_model + ".progress", "w")
        progress_file.write("# Learning step\tLearning rate\tBatch size\tEpoch\tValidation loss\tBest loss\n")
        progress_file.flush()

        self.parameters_mean = None
        self.parameters_std = None
        self.features_mean = None
        self.features_std = None

        for dataset in training_data:
            with dataset:
                parameters, features = dataset.read_data()

            m = ~np.logical_or(np.any(np.isnan(parameters), axis = 1), np.any(np.isnan(features), axis = 1))
            parameters = parameters[m,:]
            features = features[m,:]

            if training_parameters is None:
                training_parameters = parameters
                training_features = features
            else:
                training_parameters = np.concatenate((training_parameters, parameters))
                training_features = np.concatenate((training_features, features))

        self.parameters_mean = np.nanmean(training_parameters, axis = 0)
        self.parameters_std = np.nanstd(training_parameters, axis = 0)
        self.features_mean = np.nanmean(training_features, axis = 0)
        self.features_std = np.nanstd(training_features, axis = 0)

        # input parameters mean and std
        self.parameters_mean = tf.constant(self.parameters_mean, dtype=dtype, name='parameters_mean')
        self.parameters_std = tf.constant(self.parameters_std, dtype=dtype, name='parameters_std')

        # (log)-spectra mean and std
        self.features_mean = tf.constant(self.features_mean, dtype=dtype, name='features_mean')
        self.features_std = tf.constant(self.features_std, dtype=dtype, name='features_std')

        # training/validation split
        n_samples = training_parameters.shape[0]
        n_validation = int(n_samples * validation_split)
        n_training = int(n_samples) - n_validation

        training_parameters = tf.convert_to_tensor(training_parameters, dtype=dtype)
        training_features = tf.convert_to_tensor(training_features, dtype=dtype)

        # train using cooling/heating schedule for lr/batch-size
        for i in range(len(learning_rates)):

            print('learning rate = ' + str(learning_rates[i]) + ', batch size = ' + str(batch_sizes[i]))

            # set learning rate
            self.optimizer.lr = learning_rates[i]

            # Split the data into a training and validation dataset, and batch them.
            split = tf.random.shuffle([True] * n_training + [False] * n_validation)

            training_data = tf.data.Dataset.from_tensor_slices((training_parameters[split], training_features[split])).shuffle(n_training).batch(batch_sizes[i])
            validation_parameters = training_parameters[~split]
            validation_features = training_features[~split]

            # set up training loss
            training_loss = [np.infty]
            validation_loss = [np.infty]
            best_loss = np.infty
            early_stopping_counter = 0

            # loop over epochs
            with trange(max_epochs[i]) as t:
                for epoch in t:
                    # loop over batches
                    for theta, feats in training_data:

                        # training step: check whether to accumulate gradients or not (only worth doing this for very large batch sizes)
                        if gradient_accumulation_steps[i] == 1:
                            loss = self.training_step(theta, feats)
                        else:
                            loss = self.training_step_with_accumulated_gradients(theta, feats, accumulation_steps=gradient_accumulation_steps[i])

                    vloss = self.compute_loss(validation_parameters, validation_features).numpy()
                    validation_loss.append(vloss)

                    # early stopping condition
                    if vloss < best_loss:
                        best_loss = vloss
                        early_stopping_counter = 0
                    else:
                        early_stopping_counter += 1

                    # update the progressbar
                    t.set_postfix(loss=vloss, stuck=early_stopping_counter)

                    progress_file.write(f"{i}\t{learning_rates[i]:e}\t{batch_sizes[i]:d}\t{epoch:d}\t{vloss:f}\t{best_loss:f}\n")
                    progress_file.flush()

                    if early_stopping_counter >= patience_values[i]:
                        self.update_emulator_parameters()
                        self.save(filename_saved_model)
                        break

                self.update_emulator_parameters()
                self.save(filename_saved_model)
                if early_stopping_counter >= patience_values[i]:
                    print(f'Model reached early stopping condition. Validation loss = {best_loss:f}')
                else:
                    print(f'Reached max number of epochs. Validation loss = {best_loss:f}')
                print('Model saved.')

        progress_file.close()
