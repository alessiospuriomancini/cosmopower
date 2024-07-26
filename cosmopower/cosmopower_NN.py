#!/usr/bin/env python
# Author: Alessio Spurio Mancini

import numpy as np
import tensorflow as tf
import pickle
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
    """

    def __init__(self, 
                 parameters=None, 
                 modes=None, 
                 parameters_mean=None, 
                 parameters_std=None, 
                 features_mean=None, 
                 features_std=None, 
                 n_hidden=[512,512,512], 
                 restore=False, 
                 restore_filename=None, 
                 trainable=True,
                 optimizer=None,
                 verbose=False, 
                 ):
        """
        Constructor
        """
        # super
        super(cosmopower_NN, self).__init__()

        # restore
        if restore is True:
            self.restore(restore_filename)

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
            self.b.append(tf.Variable(tf.zeros([self.architecture[i+1]]), name = "b_" + str(i), trainable=trainable))
        for i in range(self.n_layers-1):
            self.alphas.append(tf.Variable(tf.random.normal([self.architecture[i+1]]), name = "alphas_" + str(i), trainable=trainable))
            self.betas.append(tf.Variable(tf.random.normal([self.architecture[i+1]]), name = "betas_" + str(i), trainable=trainable))

        # restore weights if restore = True
        if restore is True:
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
                            f"with {list(self.n_hidden)} nodes, respectively. \n"
            print(multiline_str)


# ========== TENSORFLOW implementation ===============

    # non-linear activation function
    def activation(self, 
                   x, 
                   alpha, 
                   beta
                   ):
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
        return tf.multiply(tf.add(beta, tf.multiply(tf.sigmoid(tf.multiply(alpha, x)), tf.subtract(1.0, beta)) ), x)


    # tensor predictions
    @tf.function
    def predictions_tf(self, 
                       parameters_tensor
                       ):
        r"""
        Prediction given tensor of input parameters,
        fully implemented in TensorFlow

        Parameters:
            parameters_tensor (Tensor):
                input parameters

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
                           parameters_tensor
                           ):
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

    # tensor rescale predictions
    @tf.function
    def rescaled_predictions_tf(self, 
                           parameters_tensor
                           ):
        r"""
        10^predictions given tensor of input parameters,
        fully implemented in TensorFlow. It raises 10 to the output
        of ``predictions_tf``

        Parameters:
            parameters_tensor (Tensor):
                input parameters

        Returns:
            Tensor:
                output predictions * scaling_division + scaling_subtraction
        """
        return self.postprocessing_tf(self.predictions_tf(parameters_tensor),self.processing_vectors_tf)
       
    # tensor 10.**rescaled predictions
    @tf.function
    def ten_to_rescaled_predictions_tf(self, 
                           parameters_tensor
                           ):
        r"""
        10^predictions given tensor of input parameters,
        fully implemented in TensorFlow. It raises 10 to the output
        of ``rescale_predictions_tf``

        Parameters:
            parameters_tensor (Tensor):
                input parameters

        Returns:
            Tensor:
                10^output rescaled predictions
        """
        return tf.pow(10., self.rescaled_predictions_tf(parameters_tensor)) 



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
        


    # save
    def save(self, 
             filename
             ):
        r"""
        Save network parameters

        Parameters:
            filename (str):
                filename tag (without suffix) where model will be saved
        """
        # attributes
        attributes = [self.W_, 
                      self.b_, 
                      self.alphas_, 
                      self.betas_, 
                      self.parameters_mean_, 
                      self.parameters_std_,
                      self.features_mean_,
                      self.features_std_,
                      self.n_parameters,
                      self.parameters,
                      self.n_modes,
                      self.modes,
                      self.n_hidden,
                      self.n_layers,
                      self.architecture]

        # save attributes to file
        with open(filename + ".pkl", 'wb') as f:
            pickle.dump(attributes, f)


    # restore attributes
    def restore(self, 
                filename
                ):
        r"""
        Load pre-trained model

        Parameters:
            filename (str):
                filename tag (without suffix) where model was saved
        """
        # load attributes
        with open(filename + ".pkl", 'rb') as f:
            self.W_, self.b_, self.alphas_, self.betas_, \
            self.parameters_mean_, self.parameters_std_, \
            self.features_mean_, self.features_std_, \
            self.n_parameters, self.parameters, \
            self.n_modes, self.modes, \
            self.n_hidden, self.n_layers, self.architecture = pickle.load(f)


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
        if self.parameters is not None:
            return np.stack([input_dict[k] for k in self.parameters], axis=1)
        else:
            return np.stack([input_dict[k] for k in input_dict], axis=1)


    # forward prediction given input parameters implemented in Numpy
    def forward_pass_np(self, 
                        parameters_arr
                        ):
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
                       parameters_dict
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
                            parameters_dict
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


    # Numpy array 10.**predictions
    def rescaled_predictions_np(self,
                            parameters_dict
                            ):
        r"""
        resclaing of the predictions given input parameters collected in a dict.
        Fully implemented in Numpy. It raises 10 to the output
        from ``forward_pass_np``

        Parameters:
            parameters_dict (dict [numpy.ndarray]):
                dictionary of (arrays of) parameters

        Returns:
            numpy.ndarray:
                output predictions * scaling_division + scaling_subtraction
        """
        
        return self.postprocessing_np(self.predictions_np(parameters_dict),self.processing_vectors_np)

    
    # Numpy array 10.**rescaled predictions
    def ten_to_rescaled_predictions_np(self,
                            parameters_dict
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
                10^output rescaled predictions
        """
        return 10.**self.rescaled_predictions_np(parameters_dict)


    ### Infrastructure for network training ###

    @tf.function
    def compute_loss(self,
                     training_parameters,
                     training_features
                     ):
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
                                   training_features
                                   ):
        r"""
        Computes mean squared difference and gradients

        Parameters:
            training_parameters (Tensor):
                input parameters
            training_features (Tensor):
                true features

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
              training_parameters,
              training_features,
              filename_saved_model,
              preprocessing=None,
              postprocessing_np=None,
              postprocessing_tf=None,
              processing_vectors={},
              # cooling schedule
              validation_split=0.1,
              learning_rates=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
              batch_sizes=[1024, 1024, 1024, 1024, 1024],
              gradient_accumulation_steps = [1, 1, 1, 1, 1],
              # early stopping set up
              patience_values = [100, 100, 100, 100, 100],
              max_epochs = [1000, 1000, 1000, 1000, 1000],
             ):
        r"""
        Train the model

        Parameters:
            training_parameters (dict [numpy.ndarray]):
                input parameters
            training_features (numpy.ndarray):
                true features for training
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

        # from dict to array
        training_parameters = self.dict_to_ordered_arr_np(training_parameters)

        # parameters standardisation
        self.parameters_mean = np.mean(training_parameters, axis=0)
        self.parameters_std = np.std(training_parameters, axis=0)

        # input parameters mean and std
        self.parameters_mean = tf.constant(self.parameters_mean, dtype=dtype, name='parameters_mean')
        self.parameters_std = tf.constant(self.parameters_std, dtype=dtype, name='parameters_std')
        
        
        # features scaling
        if(preprocessing=='mean_sigma'):
            processing_vectors = {'mean':np.mean(training_features,axis=0), 'sigma':np.std(training_features,axis=0)}
            
            def preprocessing(features,processing_vectors):
                return (features-processing_vectors['mean'])/processing_vectors['sigma']
            
            def postprocessing_np(features,processing_vectors):
                return features*processing_vectors['sigma'] + processing_vectors['mean']
            
            @tf.function
            def postprocessing_tf(features,processing_vectors):
                return tf.add(tf.multiply(features, processing_vectors['sigma']), processing_vectors['mean'])
               
            
        if(preprocessing=='min_max'):
            processing_vectors = {'min':np.min(training_features,axis=0), 'max':np.max(training_features,axis=0)}
            
            def preprocessing(features,processing_vectors):
                return (features-processing_vectors['min'])/(processing_vectors['max']-processing_vectors['min'])
            
            def postprocessing_np(features,processing_vectors):
                return features*(processing_vectors['max']-processing_vectors['min']) + processing_vectors['min']
            
            @tf.function
            def postprocessing_tf(features,processing_vectors):
                return tf.add(tf.multiply(features, tf.subtract(processing_vectors['max'],processing_vectors['min'])), processing_vectors['min'])
            
            
        if(preprocessing==None):
            def preprocessing(features,preprocessing):
                return features
            
            def postprocessing_np(features,preprocessing):
                return features
        
            @tf.function
            def postprocessing_tf(features,preprocessing):
                return features
            
        
        self.preprocessing = preprocessing
        self.postprocessing_np = postprocessing_np
        self.postprocessing_tf = postprocessing_tf
        
        self.processing_vectors_np = processing_vectors
        self.processing_vectors_tf = {}
        if(preprocessing is not None):
            for name in list(processing_vectors.keys()):
                self.processing_vectors_tf[name] = tf.constant(self.processing_vectors_np[name], dtype=dtype, name=name)
                
        training_features = self.preprocessing(training_features,self.processing_vectors_np)
        
        # features standardisation
        self.features_mean = np.mean(training_features, axis=0)
        self.features_std = np.std(training_features, axis=0)
        
        # (log)-spectra mean and std
        self.features_mean = tf.constant(self.features_mean, dtype=dtype, name='features_mean')
        self.features_std = tf.constant(self.features_std, dtype=dtype, name='features_std')

        # training/validation split
        n_validation = int(training_parameters.shape[0] * validation_split)
        n_training = training_parameters.shape[0] - n_validation

        # casting
        training_parameters = tf.convert_to_tensor(training_parameters, dtype=dtype)
        training_features = tf.convert_to_tensor(training_features, dtype=dtype)

        # train using cooling/heating schedule for lr/batch-size
        for i in range(len(learning_rates)):

            print('learning rate = ' + str(learning_rates[i]) + ', batch size = ' + str(batch_sizes[i]))

            # set learning rate
            self.optimizer.lr = learning_rates[i]

            # split into validation and training sub-sets
            training_selection = tf.random.shuffle([True] * n_training + [False] * n_validation)

            # create iterable dataset (given batch size)
            training_data = tf.data.Dataset.from_tensor_slices((training_parameters[training_selection], training_features[training_selection])).shuffle(n_training).batch(batch_sizes[i])

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

                    # compute validation loss at the end of the epoch
                    validation_loss.append(self.compute_loss(training_parameters[~training_selection], training_features[~training_selection]).numpy())

                    # update the progressbar
                    t.set_postfix(loss=validation_loss[-1])

                    # early stopping condition
                    if validation_loss[-1] < best_loss:
                        best_loss = validation_loss[-1]
                        early_stopping_counter = 0
                    else:
                        early_stopping_counter += 1
                    if early_stopping_counter >= patience_values[i]:
                        self.update_emulator_parameters()
                        self.save(filename_saved_model)
                        print('Validation loss = ' + str(best_loss))
                        print('Model saved.')
                        break
                self.update_emulator_parameters()
                self.save(filename_saved_model)
                print('Reached max number of epochs. Validation loss = ' + str(best_loss))
                print('Model saved.')
