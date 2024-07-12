"""
A summarizer that parameterize cosmic variance from the training set, and 
apply the cosmic variance to an ensemble point estimate, and produce the 
redshift distribution of that ensemble. 

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




TEENY = 1.0e-15

def fraction_nz(lognorm_parameters, breaks): 
    """
    Compute the integral of a log-normal distribution PDF between breaks
    lognorm_parameter should have 3 parameters: (s, loc, scale)
    the output will have size of len(breaks)-1.
    """
    cdf_right_edge = lognorm.cdf(breaks[1:], lognorm_parameters[0], lognorm_parameters[1], lognorm_parameters[2])
    cdf_left_edge = lognorm.cdf(breaks[:-1], lognorm_parameters[0], lognorm_parameters[1], lognorm_parameters[2])
    frac_list = cdf_right_edge - cdf_left_edge
    return frac_list

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


class CosmicVarianceStackInformer(PzInformer):
    """
    Informer for the cosmic variance summarizer
    The Informer is responsible for converting the training set
    into a model that describes the cosmic variance of the training set.
    """

    name = "CosmicVarianceStackInformer"
    config_options = PzInformer.config_options.copy()
    config_options.update(
        zmin=Param(float, 0.0, msg="The minimum redshift of the z grid"),
        zmax=Param(float, 3.0, msg="The maximum redshift of the z grid"),
        nzbins=Param(int, 301, msg="The number of gridpoints in the z grid"),
        varN_N_filename = Param(str, "varN_N_data.txt", msg="var N / N for the training set"),
        redshift_col=SHARED_PARAMS,
        )
    inputs = [("input", TableHandle)]
    outputs = [("model", ModelHandle)]

    def __init__(self, args, comm=None):
        PzInformer.__init__(self, args, comm=comm)

    def model_varN_overN(self, amp, gamma):
        """
        parametric model of the varN/N for the training set distribution
        self.nz_model. Parameterized by amp and gamma. 
        The first term stands for Poisson distribution, 
        The second term account for the clustering effect. 
        """
        return 1. + self.nz_model*(self.midpoints/amp)**gamma

    def loss(self,vec): 
        """
        Loss function for fitting the model_varN_overN() function 
        to the self.varN_overN. Current the default is to set self.varN_overN
        to the values given in Sanchez et al. 2015. 
        """
        amp, gamma = vec
        return np.sum((self.varN_overN - self.model_varN_overN(amp, gamma))**2)

    def run(self):
        # input_data = self.get_handle("input", allow_missing=True)

        try:
            self.config.hdf5_groupname
            input_data = self.get_data('input')[self.config.hdf5_groupname]
        except Exception:
            self.config.hdf5_groupname = None
            input_data = self.get_data('input')
        # set the zgrid values
        self.zgrid_breaks = np.linspace(self.config.zmin, self.config.zmax, self.config.nzbins)
        self.zgrid_mid = convert_breaks_to_mids(self.zgrid_breaks)
        self.midpoints = self.zgrid_mid
        
        # read the config
        specz_name = self.config.redshift_col
        varN_N_filename = self.config.varN_N_filename

        # get the spec-z from the training set, and fit a lognorm model to it
        specz = input_data[specz_name]
        lognorm_model = lognorm.fit(specz)

        breaks = convert_mids_to_breaks(self.midpoints)
        
        # nz model of the training set
        self.nz_model = fraction_nz(lognorm_model, self.zgrid_breaks)/np.sum(fraction_nz(lognorm_model, self.zgrid_breaks))*np.sum(np.histogram(specz, self.zgrid_breaks)[0])

        # read sanchez et al. 2015 from the input file, and interpolate
        sanchez_2015_varN_data = np.loadtxt(varN_N_filename)
        self.varN_overN = InterpolatedUnivariateSpline(sanchez_2015_varN_data[0],sanchez_2015_varN_data[1])(self.midpoints)

        # fit for amp and gamma
        res = minimize(self.loss, x0=[0.01, -1], method='nelder-mead')
        amp = res['x'][0]
        gamma = res['x'][1]
    
        # put the results into a model file. 
        self.model = {"nz_model":self.nz_model, "amp":amp, "gamma":gamma, "midpoints": self.midpoints}

        self.add_data('model', self.model)



class CosmicVarianceStackSummarizer(PZSummarizer):
    """
    Cosmic Variance Included Summarizer
    
    The summarizer take a model that contained the cosmic variance information for the training set, 
    the photometric catalog, and produce a n(z) distribution that include the cosmic variance. 
    """

    name = "CosmicVarianceStackSummarizer"
    config_options = PZSummarizer.config_options.copy()
    config_options.update(
        zmin=Param(float, 0.0, msg="The minimum redshift of the z grid"),
        zmax=Param(float, 3.0, msg="The maximum redshift of the z grid"),
        nzbins=Param(int, 301, msg="The number of gridpoints in the z grid"),
        ancil_type = Param(str, "zmean", msg="Type of point estimate used for histogram"),
        )
    inputs = [("input", QPHandle), ("model", ModelHandle)]
    outputs = [("output", QPHandle)]

    def __init__(self, args, comm=None):
        PZSummarizer.__init__(self, args, comm=comm)

    def rebin(self, breaks_new): 
        """
        Rebin the sampled n(z) into the grdi defined in this module. 
        """
        list_rebinned = []
        midpoints_new = breaks_new[:-1] + (breaks_new[1]-breaks_new[0])/2.
        for el in self.pz: 
            model = rv_histogram((el, convert_mids_to_breaks(self.midpoints)))
            rebinned = model.cdf(breaks_new[1:]) - model.cdf(breaks_new[:-1]) 
            list_rebinned.append(rebinned/np.trapz(rebinned, midpoints_new))
        list_rebinned = np.array(list_rebinned)
        return list_rebinned
    
    def error(self, expect, mids, num_samples = 1000): 
        """
        produce realizations of sample n(z) using the log-normal distribution
        """
        var = (self.model_coeff_variation(mids) * expect)**2
        mu = np.log(expect**2/np.sqrt(var + expect**2))
        sig_2 = np.log(var/expect**2 + 1.)
        samples = multivariate_normal.rvs(mu, np.diag(sig_2), size=1000)
        pz = np.exp(samples)
        pz = np.array([el/np.trapz(el, mids) for el in pz])
        return pz, mu, np.diag(sig_2)

    
    def summarize(self, input_data, model):
        # read the model
        self.set_data("model", model)
        model = self.get_data('model')
        
        self.nz_model = model["nz_model"]
        self.amp = model["amp"]
        self.gamma = model["gamma"]
        self.midpoints = model["midpoints"]
        self.breaks = convert_mids_to_breaks(self.midpoints)
        # set the photometric data
        self.set_data("input", input_data)
        self.run()
        self.finalize()
        return self.get_handle("output")
    
    def run(self):
        input_data = self.get_data('input')
        # get the point estimate of the photo-z method
        point_est = input_data.ancil[self.config.ancil_type]
        # bin the point estimate into thin tomographic bins
        tomographic_binning_dnnz = np.histogram(point_est, bins = self.config.nzbins, range = (self.config.zmin,self.config.zmax))
        # nz: histogram value, bins: bin edges, norm: normalized histogram value
        nz = tomographic_binning_dnnz[0]
        bins = (tomographic_binning_dnnz[1][1:] + tomographic_binning_dnnz[1][:-1])/2
        self.num_tot = np.trapz(nz, bins)
        
        # best fit var(N)/N fitted with the informer, interpolate with midpoints defined in this module
        self.model_varN_overN = 1. + self.nz_model*(self.midpoints/self.amp)**self.gamma
        self.model_coeff_variation = InterpolatedUnivariateSpline(self.midpoints, np.sqrt(self.model_varN_overN/self.nz_model), 
                                                            ext=3, k=1)
        
        # produce stack n(z) interpolation, and evaluate at the midpoints
        model_pz_stacked = InterpolatedUnivariateSpline(bins, nz, k=1, ext=3)
        pz_stacked_numgal = model_pz_stacked(self.midpoints)
        # compute the n(z) realization of the sample, pz: realizations, mu: mean values, cov: covariance of the pz
        self.expect = pz_stacked_numgal/np.sum(pz_stacked_numgal) * self.num_tot
        self.pz, self.mu, self.cov = self.error(self.expect+TEENY, self.midpoints)
        # if rebin, rebin the pz into , currently obsolete
        # sample_nz = self.rebin(self.breaks)
        
        # put the results into a qp ensemble and save
        nzs = qp.Ensemble(qp.interp, data=dict(xvals=self.midpoints, yvals=self.pz))
        self.add_data('output', nzs)
        