"""
A summarizer that parameterize cosmic variance from the training set, and 
apply the cosmic variance to an ensemble point estimate, and produce the 
redshift distribution of that ensemble. 
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
from scipy.stats import multivariate_normal
from scipy.stats import rv_histogram




TEENY = 1.0e-15

def fraction_nz(lognorm_parameters, breaks): 
    frac_list = []
    for i in range(len(breaks)-1): 
        frac_list.append(lognorm.cdf(breaks[i+1], lognorm_parameters[0], lognorm_parameters[1], lognorm_parameters[2]) - 
                        lognorm.cdf(breaks[i], lognorm_parameters[0], lognorm_parameters[1], lognorm_parameters[2]))
    frac_list = np.array(frac_list)
    return frac_list

def convert_mids_to_breaks(mids): 
    breaks = np.copy(mids)
    delta_z = breaks[1]-breaks[0]
    breaks = breaks - delta_z/2.
    breaks = breaks.tolist() 
    breaks.append(breaks[-1] + delta_z)
    breaks = np.array(breaks)
    return breaks

def convert_breaks_to_mids(breaks): 
    mids = breaks[:-1]
    delta = 0.5*(breaks[1] - breaks[0])
    mids = mids + delta
    return mids


class CosmicVarianceStackInformer(PzInformer):
    """Placeholder Informer"""

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
        return 1. + self.nz_model*(self.midpoints/amp)**gamma

    def loss(self,vec): 
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

        self.zgrid_breaks = np.linspace(self.config.zmin, self.config.zmax, self.config.nzbins)
        self.zgrid_mid = convert_breaks_to_mids(self.zgrid_breaks)

        specz_name = self.config.redshift_col
        varN_N_filename = self.config.varN_N_filename
        self.midpoints = self.zgrid_mid

        specz = input_data[specz_name]
        
        lognorm_model = lognorm.fit(specz)

        breaks = convert_mids_to_breaks(self.midpoints)

        self.nz_model = fraction_nz(lognorm_model, breaks)/np.sum(fraction_nz(lognorm_model, breaks))*np.sum(np.histogram(specz, breaks)[0])

        sanchez_2015_varN_data = np.loadtxt(varN_N_filename)
        self.varN_overN = InterpolatedUnivariateSpline(sanchez_2015_varN_data[0],sanchez_2015_varN_data[1])(self.midpoints)

        res = minimize(self.loss, x0=[0.01, -1], method='nelder-mead')
        amp = res['x'][0]
        gamma = res['x'][1]

        self.model = {"nz_model":self.nz_model, "amp":amp, "gamma":gamma, "midpoints": self.midpoints}

        self.add_data('model', self.model)



class CosmicVarianceStackSummarizer(PZSummarizer):
    """Cosmic Variance Included Summarizer"""

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

    def model_varN_overN(self, amp, gamma):
        return 1. + self.nz_model*(self.midpoints/amp)**gamma

    def loss(self,vec): 
        amp, gamma = vec
        return np.sum((self.varN_overN - self.model_varN_overN(amp, gamma))**2)
        
    def rebin(self, breaks_new): 
        list_rebinned = []
        midpoints_new = breaks_new[:-1] + (breaks_new[1]-breaks_new[0])/2.
        for el in self.pz: 
            model = rv_histogram((el, convert_mids_to_breaks(self.midpoints)))
            rebinned = np.array([model.cdf(breaks_new[i+1]) - model.cdf(breaks_new[i]) for i in range(len(breaks_new)-1)])
            list_rebinned.append(rebinned/np.trapz(rebinned, midpoints_new))
        list_rebinned = np.array(list_rebinned)
        return list_rebinned
    
    def error(self, expect, mids, num_samples = 1000): 
        var = (self.model_coeff_variation(mids) * expect)**2
        mu = np.log(expect**2/np.sqrt(var + expect**2))
        sig_2 = np.log(var/expect**2 + 1.)
        samples = multivariate_normal.rvs(mu, np.diag(sig_2), size=1000)
        pz = np.exp(samples)
        pz = np.array([el/np.trapz(el, mids) for el in pz])
        return pz, mu, np.diag(sig_2)

    
    def summarize(self, input_data, model):
        
        self.set_data("model", model)
        model = self.get_data('model')
        
        self.nz_model = model["nz_model"]
        self.amp = model["amp"]
        self.gamma = model["gamma"]
        self.midpoints = model["midpoints"]
        self.breaks = convert_mids_to_breaks(self.midpoints)

        self.set_data("input", input_data)
        self.run()
        self.finalize()
        return self.get_handle("output")
    
    def run(self):
        input_data = self.get_data('input')

        point_est = input_data.ancil[self.config.ancil_type]

        tomographic_binning_dnnz = np.histogram(point_est, bins = self.config.nzbins, range = (self.config.zmin,self.config.zmax))

        nz = tomographic_binning_dnnz[0]
        bins = (tomographic_binning_dnnz[1][1:] + tomographic_binning_dnnz[1][:-1])/2
        norm = nz/np.trapz(nz,bins)

        self.model_varN_overN = 1. + self.nz_model*(self.midpoints/self.amp)**self.gamma
        model_coeff_variation = InterpolatedUnivariateSpline(self.midpoints, np.sqrt(self.model_varN_overN/self.nz_model), 
                                                            ext=3, k=1)
        
        
        self.model_coeff_variation = model_coeff_variation
        self.num_tot = np.trapz(nz, bins)
        model_pz_stacked = InterpolatedUnivariateSpline(bins, nz, k=1, ext=3)
        pz_stacked_numgal = model_pz_stacked(self.midpoints)
        self.expect = pz_stacked_numgal/np.sum(pz_stacked_numgal) * self.num_tot
        self.pz, self.mu, self.cov = self.error(self.expect, self.midpoints)
        
        sample_nz = self.rebin(self.breaks)
        
        nzs = qp.Ensemble(qp.interp, data=dict(xvals=self.midpoints, yvals=sample_nz))
        
        self.add_data('output', nzs)
        
        
        
        
        
        
        
        
        
        
        






        