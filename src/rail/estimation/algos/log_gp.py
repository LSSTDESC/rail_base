"""
A summarizer that takes the input of a photo-z point estimate qp ensemble, and the cluster redshift likelihood
and run logistic Gaussian process to estimate the posteroir of the redshift distribution n(z) for a sample. 

Author: Markus Michael Rau, Tianqing Zhang
"""


import numpy as np
import qp
from ceci.config import StageParameter as Param
from rail.estimation.summarizer import PZSummarizer
from rail.estimation.informer import PzInformer
from rail.core.data import QPHandle, TableHandle, ModelHandle, Hdf5Handle
from rail.core.common_params import SHARED_PARAMS

from scipy.stats import lognorm
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import minimize
from scipy.stats import multivariate_normal, rv_histogram
from scipy.integrate import simps
from typing import Callable




class EllipticalSliceSampler:
    def __init__(self, prior_mean: np.ndarray,
                 prior_cov: np.ndarray,
                 loglik: Callable):
        """
        Initialize the Elliptical Slice Sampler.

        Parameters:
        prior_mean (np.ndarray): Mean of the prior distribution.
        prior_cov (np.ndarray): Covariance matrix of the prior distribution.
        loglik (Callable): Log-likelihood function.
        """
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov

        self.loglik = loglik

        self._n = len(prior_mean)  # Dimensionality of the parameter space
        self._chol = np.linalg.cholesky(prior_cov)  # Cholesky decomposition of the prior covariance matrix

        # Initialize state with a sample from the prior distribution
        self._state_f = self._chol @ np.random.randn(self._n) + prior_mean

    def _indiv_sample(self):
        """Main algorithm for generating individual samples."""
        f = self._state_f  # previous cached state
        nu = self._chol @ np.random.randn(self._n)  # choose ellipse using prior
        log_y = self.loglik(f) + np.log(np.random.uniform())  # ll threshold
        
        theta = np.random.uniform(0., 2*np.pi)  # initial proposal
        theta_min, theta_max = theta-2*np.pi, theta  # define bracket

        # main loop:  accept sample on bracket, else shrink bracket and try again
        while True:  
            assert theta != 0
            f_prime = (f - self.prior_mean)*np.cos(theta) + nu*np.sin(theta)
            f_prime += self.prior_mean
            if self.loglik(f_prime) > log_y:  # accept
                self._state_f = f_prime
                return
            
            else:  # shrink bracket and try new point
                if theta < 0:
                    theta_min = theta
                else:
                    theta_max = theta
                theta = np.random.uniform(theta_min, theta_max)

    def sample(self,
               n_samples: int,
               n_burn: int = 500) -> np.ndarray:
        """
        Generate samples from the posterior distribution.

        Parameters:
        n_samples (int): Number of samples to generate.
        n_burn (int): Number of burn-in samples to discard. Default is 500.

        Returns:
        np.ndarray: Array of generated samples.
        """
        
        samples = []
        for i in range(n_samples):
            self._indiv_sample()
            if i > n_burn: # Discard burn-in samples
                samples.append(self._state_f.copy())

        return np.stack(samples)
    

def convert_s_to_nz(s): 
    """
    Convert a vector of log-probability to probabilities using the softmax function.

    Parameters:
    s (array-like): Input array of log-probability.

    Returns:
    np.ndarray: Array of probabilities.
    """
    nz = np.array([np.exp(el)/np.sum(np.exp(s)) for el in s])
    return nz


class LogLike(object): 
    
    def __init__(self, zmid_wx, mean_wx, cov_wx): 
        """
        Initialize the LogLike object.

        Parameters:
        zmid_wx (array-like): Midpoints of redshift bins where cluster redshift is defined.
        mean_wx (array-like): Mean vector for the cluster redshift distribution.
        cov_wx (array-like): Covariance matrix for the cluster redshift distribution.
        """
        self.zmid_wx = zmid_wx
        self.mean_wx = mean_wx
        self.cov_wx = cov_wx
        
    def loglike_svec_given_amp(self, zmid, amp):
        def loss(s_vec): 
            s_vec_wx = InterpolatedUnivariateSpline(zmid,s_vec)(self.zmid_wx)
            nz_relevant = convert_s_to_nz(s_vec_wx)
            return multivariate_normal.logpdf(nz_relevant * amp, self.mean_wx, self.cov_wx) #+ multivariate_normal.logpdf(s_vec, self.mean_dnnz, self.cov_dnnz)
        return loss
               
    def loglike_amp_given_svec(self, zmid, s_vec): 
        def loss(amp): 
            s_vec_wx = InterpolatedUnivariateSpline(zmid,s_vec)(self.zmid_wx)
            nz_relevant = convert_s_to_nz(s_vec_wx)
            return multivariate_normal.logpdf(nz_relevant * amp, self.mean_wx, self.cov_wx)
        return loss

def convert_mids_to_breaks(mids): 
    """
    Convert midpoints of a grid to the breaks of the grid
    Assume the grid points have equal distances
    The output will have size of len(mids)+1
    """
    breaks = np.copy(mids)
    delta_z = breaks[1]-breaks[0]
    breaks = breaks - delta_z/2.
    breaks = breaks.tolist() 
    breaks.append(breaks[-1] + delta_z)
    breaks = np.array(breaks)
    return breaks

def convert_breaks_to_mids(breaks): 
    """
    Convert midpoints of a grid to the breaks of the grid
    Assume the grid points have equal distances
    The output will have size of len(mids)+1
    """
    mids = breaks[:-1]
    delta = 0.5*(breaks[1] - breaks[0])
    mids = mids + delta
    return mids


class LogisticGPSummarizer(PZSummarizer):
    """
    Logistic Gaussian Summarizer
    
    The summarizer takes realizations of a photo-z point estimate qp ensemble, 
    and the cluster redshift likelihood in the form of Gaussian distribution, 
    perform logistic Gaussian Process and estimate the n(z) combining the two
    information. 
    """

    name = "LogisticGPSummarizer"
    config_options = PZSummarizer.config_options.copy()
    config_options.update(
        zmin=Param(float, 0.0, msg="The minimum redshift of the z grid"),
        zmax=Param(float, 3.0, msg="The maximum redshift of the z grid"),
        nzbins=Param(int, 301, msg="The number of gridpoints in the z grid"),
        n_steps=Param(int, 5000, msg="N-steps for MCMC sampling"),
        afterburner = Param(int, 2000, msg = 'Remove the samples before chain converge')
        )
    inputs = [("input", QPHandle), ("model", ModelHandle)]
    outputs = [("output", QPHandle)]

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

    
    def summarize(self, input_data, model):
        """
        Summarize the input data using the model.

        Parameters:
        input_data: Input pz distributions from photo-z methods
        model: Model containing cluster redshift information.

        Returns:
        QPHandle: Handle to the output data.
        """
        # read the model
        self.set_data("model", model)
        model = self.get_data('model')
        
        self.zmid_wx = model["zmid_wx"]
        self.signal_wx = model["signal_wx"]
        self.cov_wx = model["cov_wx"]
        # set the photometric data
        self.set_data("input", input_data)
        self.run()
        self.finalize()
        return self.get_handle("output")
    
    def sample_joint(self): 
        """
        Perform joint sampling of amplitude and s_vec using MCMC.

        Returns:
        tuple: Arrays of sampled amplitudes and s_vecs.
        """
        loglike_model = LogLike(self.zmid_wx, self.signal_wx, self.cov_wx)
        # TEENY = np.random.uniform(0,1e-16,len(self.zgrid_mid))
        trace_amp = [50.0]
        trace_svec = [np.log(self.qp_output.pdf(self.zgrid_mid)[0])]

        log_pz = np.log(self.qp_output.pdf(self.zgrid_mid))
        mean_pz = np.mean(log_pz,axis = 0)

        cov_pz = np.cov(log_pz.T)

        for step in range(self.config.n_steps): 
            #update amp 
            if step%1000 == 0:
                print("Step "+str(step))
            loss_amp_given_svec = loglike_model.loglike_amp_given_svec(self.zgrid_mid,trace_svec[-1])
            proposed_amp = np.random.normal(trace_amp[-1], 0.1)

            log_new = loss_amp_given_svec(proposed_amp)
            log_old = loss_amp_given_svec(trace_amp[-1])
            log_accept_ratio = log_new - log_old
            rnd_curr = np.log(np.random.uniform(low=0.0, high=1.0))
            if log_accept_ratio > rnd_curr:
                trace_amp.append(proposed_amp)
            else:
                trace_amp.append(trace_amp[-1])

            #update svec
            
            
            
            loss_svec_given_amp = loglike_model.loglike_svec_given_amp(self.zgrid_mid, trace_amp[-1])
            sample_svec = EllipticalSliceSampler(mean_pz,cov_pz, loss_svec_given_amp)
            
            trace_svec.append(sample_svec.sample(20, 10)[-1])
        return np.array(trace_amp), np.array(trace_svec)

    def run(self):
        """
        Execute the summarization process.
        """
        input_data = self.get_data('input')
        self.qp_output = input_data
        
        self.zgrid_breaks = np.linspace(self.config.zmin, self.config.zmax, self.config.nzbins)
        self.zgrid_mid = convert_breaks_to_mids(self.zgrid_breaks)
        
        self.trace_amp0, self.trace_svec0 = self.sample_joint()
        
        self.trace_bin0 = np.column_stack((self.trace_amp0, self.trace_svec0))
        
        self.trace_nz0 = np.array([convert_s_to_nz(el) for el in self.trace_bin0[:, 1:]])
        
        nzs = qp.Ensemble(qp.interp, data=dict(xvals=self.zgrid_mid, yvals=self.trace_nz0[self.config.afterburner:]))
        
        self.add_data('output', nzs)
        