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
from scipy.integrate import simps




class EllipticalSliceSampler:
    def __init__(self, prior_mean: np.ndarray,
                 prior_cov: np.ndarray,
                 loglik: Callable):

        self.prior_mean = prior_mean
        self.prior_cov = prior_cov

        self.loglik = loglik

        self._n = len(prior_mean)  # dimensionality
        self._chol = np.linalg.cholesky(prior_cov)  # cache cholesky

        # init state; cache prev states
        self._state_f = self._chol @ np.random.randn(self._n) + prior_mean

    def _indiv_sample(self):
        """main algo for indiv samples"""
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
        """Returns n_samples samples"""
        
        samples = []
        for i in range(n_samples):
            self._indiv_sample()
            if i > n_burn:
                samples.append(self._state_f.copy())

        return np.stack(samples)
    

def convert_s_to_nz(s): 
    nz = np.array([np.exp(el)/np.sum(np.exp(s)) for el in s])
    return nz


class LogLike(object): 
    
    def __init__(self, pz_ensemble, zmid_wx, mean_wx, cov_wx): 
        
        # self.pz_at_zmid = pz_ensemble.pdf(zmid_wx)
        # self.mu_pz = np.mean(np.log(self.pz_at_zmid), axis = 0)
        # self.cov_pz = np.cov(np.log(self.pz_at_zmid).T)
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
        n_steps=Param(int, 1000, msg="N-steps for MCMC sampling"),
        )
    inputs = [("input", QPHandle), ("model", ModelHandle)]
    outputs = [("output", QPHandle)]

    def __init__(self, args, comm=None):
        PZSummarizer.__init__(self, args, comm=comm)

    
    def summarize(self, input_data, model):
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

        loglike_model = LogLike(self.qp_output, self.zmid_wx, self.signal_wx, self.cov_wx)
        # TEENY = np.random.uniform(0,1e-16,len(self.zgrid_mid))
        trace_amp = [55.6]
        trace_svec = [np.log(self.qp_output.pdf(self.zgrid_mid)[0])]

        log_pz = np.log(self.qp_output.pdf(self.zgrid_mid))
        mean_pz = np.mean(log_pz,axis = 0)

        cov_pz = np.cov(log_pz.T)

        for step in range(self.config.n_steps): 
            # print(step)
            #update amp 
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
        
        input_data = self.get_data('input')
        self.qp_output = input_data
        
        self.zgrid_breaks = np.linspace(self.config.zmin, self.config.zmax, self.config.nzbins)
        self.zgrid_mid = convert_breaks_to_mids(self.zgrid_breaks)
        
        self.trace_amp0, self.trace_svec0 = self.sample_joint()
        
        self.trace_bin0 = np.column_stack((self.trace_amp0, self.trace_svec0))
        
        self.trace_nz0 = np.array([convert_s_to_nz(el) for el in self.trace_bin0[:, 1:]])
        
        nzs = qp.Ensemble(qp.interp, data=dict(xvals=self.zgrid_mid, yvals=self.trace_nz0))
        
        self.add_data('output', nzs)
        