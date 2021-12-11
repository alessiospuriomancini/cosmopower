#!/usr/bin/env/python
# author: Alessio Spurio Mancini

import numpy as np
import os
from scipy.io import FortranFile
import sys
import tensorflow_probability as tfp
import tensorflow as tf
import cosmopower as cp

class tf_planck2018_lite:
    '''
    TensorFlow version of Planck's plik-lite likelihood.
    Adapted from H. Prince and J. Dunkley's [planck-lite-py](https://github.com/heatherprince/planck-lite-py).
    If you use this likelihood please cite:
        - CosmoPower release paper [Spurio Mancini et al. 2021](https://arxiv.org/abs/2106.03846)
        - [Prince & Dunkley 2019](https://arxiv.org/abs/1909.05869)
        - [_Planck_ 2018 likelihood paper](https://arxiv.org/abs/1907.12875)

    Attributes
    ----------
    parameters : `list` [`str`], optional
        list of parameters varied in the analysis
    tf_planck_lite_directory_path: 'str', optional
        path to `tf_planck_lite` directory
    spectra : `str`, optional
        `TT` or `TTTEEE`, whether to perform temperature-only or joint temperature-polarisation analysis
    use_low_ell_bins : `bool`, optional
        see [Prince & Dunkley 2019](https://arxiv.org/abs/1909.05869)
    units_factor : `float`, optional
        conversion factor (T_CMB)^2, default (2.7255e6)^2
    tt_emu_model, te_emu_model, ee_emu_model : `cosmopower.cosmopower_NN` or `cosmopower.cosmopower_PCAplusNN`,
        CosmoPower emulator models
    '''

# ===== INIT ==========
    def __init__(self, 
                 parameters=None,
                 tf_planck2018_lite_path='./',
                 spectra='TTTEEE',
                 use_low_ell_bins=False,
                 units_factor=(2.7255e6)**2,
                 tt_emu_model=None,
                 te_emu_model=None,
                 ee_emu_model=None,
                 ):
        """
        Constructor
        """
        # priors
        self.parameters=parameters
        self.set_priors(self.parameters)

        # Attributes
        self.data_directory=os.path.join(tf_planck2018_lite_path,'data')
        self.spectra = spectra
        self.use_low_ell_bins=use_low_ell_bins
        self.units_factor = units_factor

        # CosmoPower emulators
        if tt_emu_model is not None:
            self.tt_emu_model = tt_emu_model 
        else:
            raise ValueError('Unrecognised emulator for TT spectrum')
        if te_emu_model is not None:
            self.te_emu_model = te_emu_model
        else:
            raise ValueError('Unrecognised emulator for TE spectrum')
        if ee_emu_model is not None:
            self.ee_emu_model = ee_emu_model 
        else:
            raise ValueError('Unrecognised emulator for EE spectrum')

        if not (self.tt_emu_model.parameters == self.te_emu_model.parameters == self.ee_emu_model.parameters):
            print('different ordering in emulators, stopping')
            return 1

        self.parameter_keys = list(self.parameters.keys())

        if self.use_low_ell_bins:
            self.nbintt_low_ell=2
            self.plmin_TT=2
        else:
            self.nbintt_low_ell=0
            self.plmin_TT=30

        self.plmin=30
        self.plmax=2508

        self.data_dir=self.data_directory+'/planck2018_plik_lite/'
        version=22

        if self.spectra=='TTTEEE':
            self.use_tt=True
            self.use_ee=True
            self.use_te=True
        else:
            print('Spectra must be TTTEEE')
            return 1

        self.nbintt_hi = 215 #30-2508   #used when getting covariance matrix
        self.nbinte = 199 #30-1996
        self.nbinee = 199 #30-1996
        self.nbin_hi=self.nbintt_hi+self.nbinte+self.nbinee

        self.nbintt=self.nbintt_hi+self.nbintt_low_ell
        self.nbin_tot=self.nbintt+self.nbinte+self.nbinee

        self.like_file = self.data_dir+'cl_cmb_plik_v'+str(version)+'.dat'
        self.cov_file  = self.data_dir+'c_matrix_plik_v'+str(version)+'.dat'
        self.blmin_file = self.data_dir+'blmin.dat'
        self.blmax_file = self.data_dir+'blmax.dat'
        self.binw_file = self.data_dir+'bweight.dat'

        self.bval, self.X_data, self.X_sig=np.genfromtxt(self.like_file, unpack=True)
        self.X_data = tf.convert_to_tensor(self.X_data.astype('float32'))
        self.X_data = tf.reshape(self.X_data, [1, tf.shape(self.X_data)[0]])

        # read in binned ell value, C(l) TT, TE and EE and errors
        # use_tt etc to select relevant parts
        self.blmin=np.loadtxt(self.blmin_file).astype(int)
        self.blmax=np.loadtxt(self.blmax_file).astype(int)
        self.bin_w=np.loadtxt(self.binw_file).astype('float32')
        self.blmin_TT=self.blmin
        self.blmax_TT=self.blmax
        self.bin_w_TT=self.bin_w

        self.fisher=self.get_inverse_covmat()

        begintt = self.blmin_TT+self.plmin_TT-2
        endtt = self.blmax_TT+self.plmin_TT+1-2
        beginte = self.blmin+self.plmin-2
        endte = self.blmax+self.plmin+1-2
        beginee = self.blmin+self.plmin-2
        endee = self.blmax+self.plmin+1-2

        beginwtt = self.blmin_TT
        endwtt = self.blmax_TT+1
        beginwte = self.blmin
        endwte = self.blmax+1
        beginwee = self.blmin
        endwee = self.blmax+1

        indicestt = []
        windowtt = []
        indices_reptt = []
        indiceste = []
        windowte = []
        indices_repte = []
        indicesee = []
        windowee = []
        indices_repee = []

        for i in range(self.nbintt):
            idxreptt = np.repeat(i, len(np.arange(begintt[i], endtt[i])))
            indicestt.append(np.arange(begintt[i], endtt[i]))
            windowtt.append(self.bin_w_TT[beginwtt[i]:endwtt[i]])
            indices_reptt.append(idxreptt)
        flat_list = [item for sublist in indices_reptt for item in sublist]
        self.indices_reptt = np.array(flat_list)
        flat_list = [item for sublist in indicestt for item in sublist]
        self.indicestt = np.array(flat_list)
        flat_list = [item for sublist in windowtt for item in sublist]
        self.windowtt = np.array(flat_list)

        for i in range(self.nbinte):
            idxrepte = np.repeat(self.nbintt+i, len(np.arange(beginte[i], endte[i])))
            indiceste.append(self.plmax-1+np.arange(beginte[i], endte[i]))
            windowte.append(self.bin_w[beginwte[i]:endwte[i]])
            indices_repte.append(idxrepte)
        flat_list = [item for sublist in indices_repte for item in sublist]
        self.indices_repte = np.array(flat_list)
        flat_list = [item for sublist in indiceste for item in sublist]
        self.indiceste = np.array(flat_list)
        flat_list = [item for sublist in windowte for item in sublist]
        self.windowte = np.array(flat_list)

        for i in range(self.nbinee):
            idxrepee = np.repeat(self.nbintt+self.nbinte+i, len(np.arange(beginee[i], endee[i])))
            indicesee.append(self.plmax-1+self.plmax-1+np.arange(beginee[i], endee[i]))
            windowee.append(self.bin_w[beginwee[i]:endwee[i]])
            indices_repee.append(idxrepee)
        flat_list = [item for sublist in indices_repee for item in sublist]
        self.indices_repee = np.array(flat_list)
        flat_list = [item for sublist in indicesee for item in sublist]
        self.indicesee = np.array(flat_list)
        flat_list = [item for sublist in windowee for item in sublist]
        self.windowee = np.array(flat_list)

        self.windowtt = tf.constant(self.windowtt, shape=[1,self.windowtt.shape[0]])
        self.windowte = tf.constant(self.windowte, shape=[1,self.windowte.shape[0]])
        self.windowee = tf.constant(self.windowee, shape=[1,self.windowee.shape[0]])
        self.indices_reptt = tf.constant(self.indices_reptt)
        self.indices_repte = tf.constant(self.indices_repte)
        self.indices_repee = tf.constant(self.indices_repee)
        self.indicestt = tf.constant(self.indicestt)
        self.indiceste = tf.constant(self.indiceste)
        self.indicesee = tf.constant(self.indicesee)

        self.indices_rep = tf.concat([self.indices_reptt, self.indices_repte, self.indices_repee], axis=0)
        self.indices = tf.concat([self.indicestt, self.indiceste, self.indicesee], axis=0)


# ===== INVERSE COVARIANCE ======
    def get_inverse_covmat(self):
        """
        Read and invert covariance matrix
        """
        #read full covmat
        f = FortranFile(self.cov_file, 'r')
        covmat = f.read_reals(dtype=float).reshape((self.nbin_hi,self.nbin_hi))
        for i in range(self.nbin_hi):
            for j in range(i,self.nbin_hi):
                covmat[i,j] = covmat[j,i]

        #select relevant covmat
        if self.use_tt and self.use_ee and self.use_te:
            #use all
            bin_no=self.nbin_hi
            cov=covmat
        else:
            print("not implemented")

        cov = tf.convert_to_tensor(cov.astype('float32'))

        fisher = tf.linalg.cholesky_solve(tf.linalg.cholesky(cov), tf.eye(bin_no))
        fisher = tf.transpose(fisher)

        return fisher


    def from_parameters_tensor_to_table(self,
                                        parameters_tensor,
                                        ):
        """
        Convert parameters `tf.Tensor` into `tf.lookup.experimental.DenseHashTable`

        Parameters
        ----------
        parameters_tensor : `tf.Tensor`

        Returns
        -------
        parameters_table : `tf.lookup.experimental.DenseHashTable`
        """
        parameters_values = tf.transpose(parameters_tensor)
        parameters_table = tf.lookup.experimental.DenseHashTable(key_dtype=tf.string, 
                                                                 value_dtype=tf.float32, 
                                                                 empty_key="<EMPTY_SENTINEL>", 
                                                                 deleted_key="<DELETE_SENTINEL>", 
                                                                 default_value=tf.zeros([parameters_tensor.shape[0]]))
        parameters_table.insert(self.parameter_keys, parameters_values)
        return parameters_table


    @tf.function
    def get_loglkl(self, parameters):
        """
        Compute log-likelihood

        Parameters
        ----------
        parameters : `tf.Tensor`

        Returns
        -------
        loglkl : `tf.Tensor`
        """
        # creating lookup table with parameters
        parameters_table = self.from_parameters_tensor_to_table(parameters)
        cal = parameters_table.lookup(tf.constant(['A_planck']))
        cosmo_params = tf.transpose(parameters_table.lookup(tf.constant(self.tt_emu_model.parameters)))

        # sourcing C_ells from CosmoPower
        Cltt= self.tt_emu_model.ten_to_predictions_tf(cosmo_params)
        Clte= self.te_emu_model.predictions_tf(cosmo_params)
        Clee= self.ee_emu_model.ten_to_predictions_tf(cosmo_params)

        # units of measure
        Cl = tf.scalar_mul(self.units_factor, tf.concat([Cltt, Clte, Clee], axis=1))

        # window function: batches
        self.window_ttteee = tf.concat([self.windowtt, self.windowte, self.windowee], axis=1)
        self.window_tile = tf.tile(self.window_ttteee, [parameters.shape[0], 1])

        # binning C_ells
        Cl_bin = tf.math.segment_sum( \
        tf.transpose( \
        tf.math.multiply(tf.gather(Cl, self.indices, axis=1), self.window_tile)), \
        self.indices_rep)

        # final theory prediction
        X_model = tf.transpose(tf.divide(Cl_bin, tf.square(cal)))

        # chi2 computation
        diff_vec = tf.subtract(self.X_data, X_model)
        chi2 = tf.matmul(self.fisher, tf.transpose(diff_vec))
        chi2 = tf.matmul(diff_vec, chi2)
        chi2 = tf.linalg.diag_part(chi2)
        loglkl = tf.scalar_mul(-0.5, chi2)
        loglkl = tf.reshape(loglkl, [parameters.shape[0], 1])

        return loglkl


# ===== LOGLKL ======
    @tf.function
    def loglkl(self, 
               parameters, 
               prodpri
               ):
        """
        Call `get_loglkl` and check for parameters beyond prior ranges

        Parameters
        ----------
        parameters : `tf.Tensor`

        prodpri : `tf.Tensor`
        Returns
        -------
        """
        loglike = self.get_loglkl(parameters)
        loglike = tf.where(tf.equal(prodpri, 0.), -0.5*2e12, loglike)
        return loglike


# ===== PRIORS ========
    def set_priors(self, 
                   parameters
                  ):
        """
        Prior distributions for all parameters

        Parameters
        ----------
        parameters : `dict`
        """
        model = []
        for elem in parameters:
                low, high, name = parameters[elem]
                if name=='uniform':
                    model.append(tfp.distributions.Uniform(low=low, high=high))
                elif name=='gaussian':
                    model.append(tfp.distributions.Normal(loc=low, scale=high))
        self.priors = tfp.distributions.Blockwise(model)


# ===== POSTERIOR ========
    @tf.function
    def posterior(self, params):
        """
        Parameters
        ----------
        params : `tf.Tensor`

        Returns
        -------
        sum_loglkl_logpr : `tf.Tensor`
        """
        pr = tf.reshape(self.priors.prob(params), [params.shape[0], 1])
        logprodPri, loglike  = tf.math.log(pr), self.loglkl(params, pr)
        logprodPri = tf.where(tf.math.is_inf(logprodPri), -1e32, logprodPri)
        sum_loglkl_logpr = tf.add(loglike, logprodPri)

        return sum_loglkl_logpr
