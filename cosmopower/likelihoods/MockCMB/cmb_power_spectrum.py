import numpy as np
import matplotlib.pyplot as plt
from cobaya.likelihood import Likelihood
from typing import Optional

class MockCMB(Likelihood):
	fsky: float = 1.0
	use_tt: bool = True
	use_te: bool = True
	use_ee: bool = True
	use_bb: bool = True
	noise_curve: Optional[str] = None
	spectrum_file: str = "cmb_spectra.txt"
	
	def initialize(self):
		import os
		path = os.path.dirname(os.path.abspath(__file__))
		data = np.loadtxt(os.path.join(path, self.spectrum_file))
		
		self.ls = data[:,0].astype(int)
		self.cltt = data[:,1]
		self.clee = data[:,2]
		self.clbb = data[:,3]
		self.clte = data[:,4]
		
		self.nltt = np.sqrt(2.0 * self.cltt ** 2.0 / (self.fsky * (2.0 * self.ls + 1.0)))
		self.nlee = np.sqrt(2.0 * self.clee ** 2.0 / (self.fsky * (2.0 * self.ls + 1.0)))
		self.nlbb = np.sqrt(2.0 * self.clbb ** 2.0 / (self.fsky * (2.0 * self.ls + 1.0)))
		
		self.nlte = np.sqrt((self.cltt * self.clee + self.clte ** 2.0) / (self.fsky * (2.0 * self.ls + 1.0)))
		
		if self.noise_curve == "SO":
			TT_ells, TT_Nell = np.loadtxt("test/SO_LAT_Nell_T_atmv1_goal_fsky0p4_ILC_CMB.txt", usecols = (0, 1), unpack = True)
			EE_ells, EE_Nell = np.loadtxt("test/SO_LAT_Nell_P_goal_fsky0p4_ILC_CMB_E.txt", usecols = (0, 1), unpack = True)
			BB_ells, BB_Nell = np.loadtxt("test/SO_LAT_Nell_P_goal_fsky0p4_ILC_CMB_B.txt", usecols = (0, 1), unpack =  True)
			
			mask = np.logical_and(self.ls >= 40, self.ls <= 6000)
			
			self.fsky = 0.4
			self.ls = self.ls[mask]
			self.cltt = self.cltt[mask]
			self.clee = self.clee[mask]
			self.clbb = self.clbb[mask]
			self.clte = self.clte[mask]
			
			TT_Nell = TT_Nell[np.logical_and(TT_ells >= 40, TT_ells <= 6000)] / (2.7255e6 ** 2.0)
			EE_Nell = EE_Nell[np.logical_and(EE_ells >= 40, EE_ells <= 6000)] / (2.7255e6 ** 2.0)
			BB_Nell = BB_Nell[np.logical_and(BB_ells >= 40, BB_ells <= 6000)] / (2.7255e6 ** 2.0)
			
			self.nltt = np.sqrt(2.0 * (self.cltt + TT_Nell) ** 2.0 / (self.fsky * (2.0 * self.ls + 1.0)))
			self.nlee = np.sqrt(2.0 * (self.clee + EE_Nell) ** 2.0 / (self.fsky * (2.0 * self.ls + 1.0)))
			self.nlbb = np.sqrt(2.0 * (self.clbb + BB_Nell) ** 2.0 / (self.fsky * (2.0 * self.ls + 1.0)))
			
			self.nlte = np.sqrt(((self.cltt + TT_Nell) * (self.clee + EE_Nell) + self.clte ** 2.0) / (self.fsky * (2.0 * self.ls + 1.0)))
			
			self.log.info("Added Simons Observatory noise curves.")
		else:
			self.log.info(f"Using cosmic variance limit with fsky = {self.fsky}")
	
	def get_requirements(self):
		return { "Cl" : { k : 6000 for k in [ "tt", "te", "ee", "bb" ] } }
	
	def logp(self, **params_values):
		Cls = self.provider.get_Cl(ell_factor = True, units = "1")
		
		chi2 = 0.0
		chis = {}
		
		if self.use_tt:
			chis["tt"] = np.sum(((self.cltt - Cls["tt"][self.ls]) / self.nltt) ** 2.0)
			chi2 += chis["tt"]
		if self.use_te:
			chis["te"] = np.sum(((self.clte - Cls["te"][self.ls]) / self.nlte) ** 2.0)
			chi2 += chis["te"]
		if self.use_ee:
			chis["ee"] = np.sum(((self.clee - Cls["ee"][self.ls]) / self.nlee) ** 2.0)
			chi2 += chis["ee"]
		if self.use_bb:
			chis["bb"] = np.sum(((self.clbb - Cls["bb"][self.ls]) / self.nlbb) ** 2.0)
			chi2 += chis["bb"]
		
		if False:
			fig, axes = plt.subplots(2, 2, figsize = (12, 8), sharex = True)
			
			axes[0,0].plot(self.ls, Cls["tt"][self.ls], c = "k", lw = 2)
			axes[0,0].plot(self.ls, self.cltt, c = "C0", ls = ":", lw = 2)
			axes[0,0].fill_between(self.ls, self.cltt - self.nltt, self.cltt + self.nltt, fc = "C0", alpha = 0.3)
			axes[0,0].semilogx()
			axes[0,0].set_xlim(self.ls.min(), self.ls.max())
			axes[0,0].semilogy()
			axes[0,0].set_title(f"chisqr = {chis['tt']:.2f}")
			
			axes[1,0].plot(self.ls, Cls["te"][self.ls], c = "k", lw = 2)
			axes[1,0].plot(self.ls, self.clte, c = "C1", ls = ":", lw = 2)
			axes[1,0].fill_between(self.ls, self.clte - self.nlte, self.clte + self.nlte, fc = "C1", alpha = 0.3)
			axes[1,0].set_title(f"chisqr = {chis['te']:.2f}")
			
			axes[0,1].plot(self.ls, Cls["ee"][self.ls], c = "k", lw = 2)
			axes[0,1].plot(self.ls, self.clee, c = "C2", ls = ":", lw = 2)
			axes[0,1].fill_between(self.ls, self.clee - self.nlee, self.clee + self.nlee, fc = "C2", alpha = 0.3)
			axes[0,1].semilogy()
			axes[0,1].set_title(f"chisqr = {chis['ee']:.2f}")
			
			axes[1,1].plot(self.ls, Cls["bb"][self.ls], c = "k", lw = 2)
			axes[1,1].plot(self.ls, self.clbb, c = "C3", ls = ":", lw = 2)
			axes[1,1].fill_between(self.ls, self.clbb - self.nlbb, self.clbb + self.nlbb, fc = "C3", alpha = 0.3)
			axes[1,1].semilogy()
			axes[1,1].set_title(f"chisqr = {chis['tt']:.2f}")
			
			plt.show()
		
		return -0.5 * chi2
