from astropy.io import fits, ascii
from astropy.table import QTable
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from stat_util import outlier_cut as OC
import matplotlib.patches as mpatches
from pcc_arrow import arrow_error, arro, pcc, arrow_error_new

from astropy import units as u
from scipy.stats import sem, binned_statistic, norm, binned_statistic_2d
import pandas as pd
import pingouin as pg
from sklearn.utils import resample

from rf_parts import boots_RFR, basic_RF, boots_RFR_updated, permutate, permutation_

from astropy.cosmology import Planck18, FlatLambdaCDM
import astropy.units as u

import argparse

import os

import pickle 

from loess.loess_2d import loess_2d

from scipy import odr

from lmfit import Model
from lmfit import Parameters, fit_report, minimize

def configure_plots():
	from matplotlib import rcParams
	"""
	Sets global Matplotlib settings to include the following features:
		- inward-facing ticks on all axis spines
		- latex rendering enabled
		- serif fonts in text
		- no frame around the legend
		- dotted gridlines (once matplotlib.pyplot.axes.grid(True)  
		- figure DPI set to 300 for PDF renders
		- default saving format as PDF
	"""
	# line settings
	rcParams['lines.linewidth'] = 2
	rcParams['lines.markersize'] = 3
	rcParams['errorbar.capsize'] = 3

	# tick settings
	rcParams['xtick.top'] = True
	rcParams['ytick.right'] = True
	rcParams['xtick.major.size'] = 7
	rcParams['xtick.minor.size'] = 4
	rcParams['xtick.direction'] = 'in'
	rcParams['ytick.major.size'] = 7
	rcParams['ytick.minor.size'] = 4
	# rcParams['ytick.major.size'] = 9
	# rcParams['ytick.minor.size'] = 6
	rcParams['ytick.direction'] = 'in'

	# text settings
	rcParams['mathtext.rm'] = 'serif'
	rcParams['font.family'] = 'serif'
	rcParams['font.size'] = 14
	#rcParams['text.usetex'] = True

	rcParams['axes.titlesize'] = 18
	rcParams['axes.labelsize'] = 15
	rcParams['axes.ymargin'] = 0.5

	# legend
	rcParams['legend.fontsize'] = 12
	rcParams['legend.frameon'] = False

	# grid in plots
	rcParams['grid.linestyle'] = ':'

	# figure settings
	#rcParams['figure.figsize'] = 5, 4
	rcParams['figure.dpi'] = 300
	rcParams['savefig.format'] = 'pdf'

	# pgf_with_latex = {
	# "text.usetex": False,            # use LaTeX to write all text
	# "pgf.rcfonts": False,           # Ignore Matplotlibrc
	# "pgf.preamble": [
	#     r'\usepackage{color}'     # xcolor for colours
	# ]}

	# rcParams.update(pgf_with_latex)


def odrfunc(x,y):
	'''Orthogonal distance regression function with residual standard deviation'''
	linear_func = lambda B, x: B[0]*x +B[1]
	linear=odr.Model(linear_func)
	mydata = odr.RealData(x, y)
	myodr = odr.ODR(mydata, linear, beta0=[1., 2.])

	myoutput = myodr.run()
	#myoutput.pprint()
	r=myoutput.beta
	rerr=myoutput.sd_beta
	re=myoutput.res_var
	
	std=np.sqrt(re)
	return r, rerr, std


def ms_randp(mass):
	'''Renzini and Pneg main sequence definition '''
	return 0.76*mass - 7.64

class Z_plots_csig(object):

	'''Contains main class for running stellar metallicities paper pipeline for SDSS galaxies
	   Reads the catalog data, performs sample selection, runs RF, performs plots using PCCS etc.
	   multiple different versions  '''

	def __init__(self, star_forming=False, SF_definition_new=True):
		''' Load data, select star-forming and passive galaxies via MS cut and sigma clip'''

		super(Z_plots_csig,self).__init__()
		# hdu=fits.open('DATA/SDSS/stellar_metallicities/james_matched_24_06_22_gas_met_environment_csigmasses.fits')[1]
		# hdu=fits.open('DATA/SDSS/stellar_metallicities/james_Zcsig_11.07.23.fits')[1]
		hdu=fits.open('data/sdss_expanded_cat_matched_14jul.fits')[1]
		# self.SFR=hdu.data['SFR']
		self.SFR=hdu.data['SFR_tot']
		# self.Mstar=hdu.data['Mstar']
		self.Mstar=hdu.data['Mass_tot']
		self.OH=hdu.data['12+log(O/H)_T04']
		self.centsat=hdu.data['CENTSAT_MASS']
		self.delta=hdu.data['OVERDENSITY']
		#self.SF=hdu.data['SF?']
		self.halo_mass=hdu.data['HALO_MASS_M']
		#self.Z=hdu.data['METAL_MW']
		self.Z=hdu.data['METAL_LW']
		sigma_v_ny=hdu.data['vdisp_nyu']
		sigma_v_mpa=hdu.data['vdisp_MPA']
		z=hdu.data['z_1']
		b_r=hdu.data['bulge_r']
		r_e=hdu.data['Rchl_r']

		phi=self.Mstar-r_e

		cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
		rapp = 1.5*u.arcsec.to('rad')*cosmo.angular_diameter_distance(z).to('kpc').value
		#b = np.where(b_r > 0)[0]
		csig = np.power(np.divide(b_r, 8*rapp), -0.04)*sigma_v_ny

		#self.csig=csig

		csig = np.log10(csig)

		#csig = 5.246 * np.log10(csig) - 3.77

		print("Median redshift is ",np.nanmedian(z))



		if SF_definition_new==False:

			idx=np.where((self.centsat==1) & (csig>0) & (csig<200) & (self.SF==star_forming))[0]
			#idx=[(self.centsat==1) & (self.SF==False)]
			#idx=np.where((self.centsat==1) & (self.halo_mass>1) )[0]
			MstarC, SFRC, OHC=self.Mstar[idx], self.SFR[idx], self.OH[idx]
			ovC,hmaC, ZC, csigC, phiC=self.delta[idx], self.halo_mass[idx], self.Z[idx], csig[idx], phi[idx]
			SFC=self.SF[idx]
			print(len(MstarC))

			idx=np.where((self.centsat==2) & (csig>0) & (csig<200) & (self.SF==star_forming))[0]
			#idx=np.where((self.centsat==2) & (self.halo_mass>1) )[0]
			#idx=[(self.centsat==2) & (self.SF==False)]
			MstarS, SFRS, OHS=self.Mstar[idx], self.SFR[idx], self.OH[idx]
			ovS,hmaS, ZS, csigS, phiS=self.delta[idx], self.halo_mass[idx], self.Z[idx], csig[idx], phi[idx]
			SFS=self.SF[idx]
			print(len(MstarS))

			idx=np.where((csig>0) & (csig<200) & (self.SF==star_forming))[0]
			MstarT, SFRT, OHT=self.Mstar[idx], self.SFR[idx], self.OH[idx]
			ran=np.random.uniform(0,1.,len(MstarT))
			ovT,hmaT, ZT, csigT, phiT=self.delta[idx], self.halo_mass[idx], self.Z[idx], csig[idx], phi[idx]
			self.dat=OC(np.array((MstarT, SFRT, OHT, ovT, hmaT, ran, ZT, csigT, phiT)))

			ran=np.random.uniform(0,1.,len(MstarC))
			self.dataC=OC(np.array((MstarC, SFRC, OHC, ovC, hmaC, ran, ZC, csigC, phiC)))

			ran=np.random.uniform(0,1.,len(MstarS))
			self.dataS=OC(np.array((MstarS, SFRS, OHS, ovS, hmaS, ran, ZS, csigS, phiS)))

		else:
			ms_sfr=ms_randp(self.Mstar)
			passive=self.SFR-ms_sfr<-0.75
			sf=self.SFR-ms_sfr>-0.75

			if star_forming==True:
				state=sf
			else:
				state=passive

			idx=np.where((self.centsat==1) & (csig<20000) & (csig>1.) & state)[0]
			#idx=[(self.centsat==1) & (self.SF==False)]
			#idx=np.where((self.centsat==1) & (self.halo_mass>1) )[0]
			MstarC, SFRC, OHC=self.Mstar[idx], self.SFR[idx], self.OH[idx]
			ovC,hmaC, ZC, csigC, phiC=self.delta[idx], self.halo_mass[idx], self.Z[idx], csig[idx], phi[idx]
			#SFC=self.SF[idx]
			print(len(MstarC))

			idx=np.where((self.centsat==2) & (csig<20000) & (csig>1.) & state)[0]
			#idx=np.where((self.centsat==2) & (self.halo_mass>1) )[0]
			#idx=[(self.centsat==2) & (self.SF==False)]
			MstarS, SFRS, OHS=self.Mstar[idx], self.SFR[idx], self.OH[idx]
			ovS,hmaS, ZS, csigS, phiS=self.delta[idx], self.halo_mass[idx], self.Z[idx], csig[idx], phi[idx]
			#SFS=self.SF[idx]
			print(len(MstarS))

			# idx=np.where((csig>0) & (csig<200) & state & (self.halo_mass>0))[0]
			idx=np.where((csig<20000) & (csig>1.) & state )[0]
			MstarT, SFRT, OHT=self.Mstar[idx], self.SFR[idx], self.OH[idx]
			ran=np.random.uniform(0,1.,len(MstarT))
			ovT,hmaT, ZT, csigT, phiT=self.delta[idx], self.halo_mass[idx], self.Z[idx], csig[idx], phi[idx]
			self.dat=OC(np.array((MstarT, SFRT, ovT, hmaT, ran, ZT, csigT, phiT)), dev=3)

			


			ran=np.random.uniform(0,1.,len(MstarC))
			self.dataC=OC(np.array((MstarC, SFRC, OHC, ovC, hmaC, ran, ZC, csigC, phiC)), dev=3)

			ran=np.random.uniform(0,1.,len(MstarS))
			self.dataS=OC(np.array((MstarS, SFRS, OHS, ovS, hmaS, ran, ZS, csigS, phiS)), dev=3)

		self.star_forming=star_forming

		if star_forming:
			self.type_='SF'
		else:
			self.type_='Q'

		print(f"Number of {self.type_} galaxies {self.dat.shape}")


	def stats(self):
		''' Show minimum and maximum values of quantities'''
		Mstar, SFR, ov, hma, ran, Z, csig, phi=self.dat
		print(f"stellar mass {min(Mstar)} {max(Mstar)}")
		print(f"sfr {min(SFR)} {max(SFR)}")
		print(f"overdensiy {min(ov)} {max(ov)}")
		print(f"halo mass {min(hma)} {max(hma)}")
		print(f"stellar Z {min(Z)} {max(Z)}")
		print(f"csig mass {min(csig)} {max(csig)}")







	def RF_separate(self,run=""):

		''' Random Forest Regression - split between centrals and satellites - save as pckl file'''

		#output_dir=f'/Users/will/OneDrive - University of Cambridge/PythonCoding/Plots/stellar_Z/data/'
		configure_plots()
		output_dir=f'/Users/will/GitHub/Stellar_Z_project/data/'
		file_name = os.path.join(output_dir, f"RF_output_summary_{self.type_}{run}_separate.pkl")


		if os.path.isfile(file_name):
			with open(os.path.join(output_dir, f"RF_output_summary_{self.type_}{run}_separate.pkl"), "rb") as input_file:
				output_summary = pickle.load(input_file)
			eout=output_summary
			#print(eout)
			performanceC=eout['performance_C']
			performanceS=eout['performance_S']
			errorsC=eout['errors_C']
			errorsS=eout['errors_S']
			loc=eout['loc']
			varnames=eout['varnames']
			# mse_test=eout['mse_test']
			# mse_train=eout['mse_train']
			# print(f"MSE_test: {mse_test}  MSE_train: {mse_train}")
			return output_summary

		else:

			MstarC, SFRC, OHC, ovC, hmaC, ran, ZC, csigC, phiC=self.dataC
			#MstarC, SFRC=MstarC+np.random.normal(0,0.2,len(MstarC)), SFRC+np.random.normal(0,0.2,len(MstarC))
			dataC=np.array([ZC, MstarC, SFRC, ovC, hmaC, ran, csigC, phiC])
			performanceC, mse_trC, mse_vaC, length=basic_RF(dataC, n_est=200, min_samp_leaf=15, max_dep=100, param_check=False) 
			performanceC, errorsC=boots_RFR_updated(dataC,  n_est=200, min_samp_leaf=15, max_dep=100, n_times=100)

			MstarS, SFRS, OHS, ovS, hmaS, ran, ZS, csigS, phiS=self.dataS
			dataS=np.array([ZS, MstarS, SFRS, ovS, hmaS, ran, csigS, phiS])

			performanceS2, mse_trS2, mse_vaS2, length=basic_RF(dataS, n_est=200, min_samp_leaf=15, max_dep=100, param_check=False) 
			performanceS, errorsS=boots_RFR_updated(dataS,  n_est=200, min_samp_leaf=15, max_dep=100, n_times=100)

			varnames = np.array([
			r'$M_*$', 
			r'SFR',  r'(1+$\delta$)',  r'$M_H$', 'R', r'$\sigma_{c}[M_{BH}]$', r'$\phi_g$'
			])
			#r'$d_{cent}$'

			idx = np.argsort(performanceC)[::-1]

			loc = np.arange(length.shape[1])

			performanceC=performanceC[idx]
			performanceS=performanceS[idx]
			errorsC=errorsC[:,idx]
			errorsS=errorsS[:,idx]
			varnames=varnames[idx]

			RF_data={}
			RF_data['performance_C']=performanceC
			RF_data['performance_S']=performanceS
			RF_data['errors_C']=errorsC
			RF_data['errors_S']=errorsS
			RF_data['loc']=loc
			RF_data['mse_train_C']=mse_trC.round(3)
			RF_data['mse_test_C']=mse_vaC.round(3)
			RF_data['mse_train_S']=mse_trS2.round(3)
			RF_data['mse_test_S']=mse_vaS2.round(3)
			RF_data['varnames']=varnames
			RF_data['idx']=idx
			output={}
			output['results']=RF_data
			file_name = os.path.join(output_dir, f"RF_output_summary_{self.type_}{run}_separate.pkl")
			if os.path.exists(file_name):
				os.remove(file_name)
			f = open(file_name, "wb")
			pickle.dump(RF_data, f, protocol=pickle.HIGHEST_PROTOCOL)
			return RF_data

		

		fig, ax = plt.subplots(figsize=(5,5))

		if self.star_forming:
			ax.text(0.6, 0.7, "Parameter importance \n in determining Z for \n"+ r'star-forming' +" galaxies", fontsize=16,
				horizontalalignment='center',  verticalalignment='center', transform=ax.transAxes)
			col='#0F2080'
			col='C10'
		else:
			ax.text(0.6, 0.7, "Parameter importance \n in determining Z for \n passive galaxies", fontsize=16,
				horizontalalignment='center',  verticalalignment='center', transform=ax.transAxes)
			col='C3'

		ax.bar(loc, performanceC, yerr=errorsC, width=0.3, 
		edgecolor=col, facecolor='w',
		hatch=2*'\\', lw=2, label='Centrals') #, \n MSE train,test = [{}, {}]'.format(mse_trC.round(3),mse_vaC.round(3)))
		ax.bar(loc+0.3, performanceS, yerr=errorsS, width=0.3, 
		edgecolor='orange', facecolor='w',
		hatch=2*'\\', lw=2, label='Satellites') #, \n MSE train,test = [{}, {}]'.format(mse_trS2.round(3),mse_vaS2.round(3)))
		ax.set_xticks(loc+0.2)
		ax.tick_params(which='both', axis='x', bottom=False, top=False)
		ax.set_ylim(0,0.7)
		ax.set_xticklabels(varnames)
		ax.set_ylabel('Relative importance')
		ax.legend()
		fig.savefig('Plots/Z_total_performance_{}_separate.pdf'.format(self.type_), bbox_inches='tight')



	def RF_combined(self, run=""):

		'''Main random forest leading to key result'''

		configure_plots()

		#output_dir=f'/Users/will/OneDrive - University of Cambridge/PythonCoding/Plots/stellar_Z/data/'
		output_dir=f'/Users/will/GitHub/Stellar_Z_project/data/'
		file_name = os.path.join(output_dir, f"RF_output_summary_{self.type_}{run}.pkl")

		if os.path.isfile(file_name):
			with open(os.path.join(output_dir, f"RF_output_summary_{self.type_}{run}.pkl"), "rb") as input_file:
				output_summary = pickle.load(input_file)
			eout=output_summary
			#print(eout)
			performance=eout['performance_sorted']
			errors=eout['errors_sorted']
			loc=eout['loc']
			varnames=eout['varnames']
			mse_test=eout['mse_test']
			mse_train=eout['mse_train']
			idx=eout['idx']
			print(f"MSE_test: {mse_test}  MSE_train: {mse_train}")
		else:

			Mstar, SFR, ov, hma, ran, Z, csig, phi=self.dat
			#MstarC, SFRC=MstarC+np.random.normal(0,0.2,len(MstarC)), SFRC+np.random.normal(0,0.2,len(MstarC))
			data=np.array([Z, Mstar, SFR, ov, hma, ran, csig, phi])
			performance, mse_tr, mse_va, length=basic_RF(data, n_est=200, min_samp_leaf=15, max_dep=100, t_size=0.5, param_check=False) 
			performance, errors=boots_RFR_updated(data,  n_est=200, min_samp_leaf=15, max_dep=100, n_times=100, t_size=0.5, wide=False)

			varnames = np.array([
			r'$M_*$', 
			r'SFR',  r'(1+$\delta$)',  r'$M_H$', 'R', r'$\sigma_{c}/M_{BH}$', r'$\phi_g$'
			])

			idx = np.argsort(performance)[::-1]

			loc = np.arange(length.shape[1])

			performance=performance[idx]
			errors=errors[:,idx]
			varnames=varnames[idx]

			RF_data={}
			RF_data['performance_sorted']=performance
			RF_data['errors_sorted']=errors
			RF_data['loc']=loc
			RF_data['mse_train']=mse_tr.round(3)
			RF_data['mse_test']=mse_va.round(3)
			RF_data['varnames']=varnames
			RF_data['idx']=idx
			output={}
			output['results']=RF_data
			file_name = os.path.join(output_dir, f"RF_output_summary_{self.type_}{run}.pkl")
			if os.path.exists(file_name):
				os.remove(file_name)
			f = open(file_name, "wb")
			pickle.dump(RF_data, f, protocol=pickle.HIGHEST_PROTOCOL)

		fig, ax = plt.subplots(figsize=(5,5))
		if self.star_forming:
			# ax.text(0.6, 0.7, "Parameter importance \n in determining Z for \n"+ r'star-forming' +" galaxies", fontsize=16,
			# 	horizontalalignment='center',  verticalalignment='center', transform=ax.transAxes)
			# ax.text(0.6, 0.7, "Parameter importance \n in determining the \n"+ 'metallicity \n Star Forming', fontsize=16,
			# 	horizontalalignment='center',  verticalalignment='center', transform=ax.transAxes)
			ax.text(0.6, 0.8, "Parameter importance \n in determining the \n metallicity \n \n", fontsize=16,
				horizontalalignment='center',  verticalalignment='center', transform=ax.transAxes)
			ax.text(0.6, 0.7, r"Star Forming", fontsize=16, color='C10',
				horizontalalignment='center',  verticalalignment='center', transform=ax.transAxes)
			col='#0F2080'
			col='C10'
		else:
			# ax.text(0.6, 0.7, "Parameter importance \n in determining Z for \n passive galaxies", fontsize=16,
			# 	horizontalalignment='center',  verticalalignment='center', transform=ax.transAxes)
			# ax.text(0.6, 0.8, "Parameter importance \n in determining the \n metallicity \n \n"+r"$\bf Passive$", fontsize=16,
			# 	horizontalalignment='center',  verticalalignment='center', transform=ax.transAxes)
			ax.text(0.6, 0.8, "Parameter importance \n in determining the \n metallicity \n \n", fontsize=16,
				horizontalalignment='center',  verticalalignment='center', transform=ax.transAxes)
			ax.text(0.6, 0.7, r"Passive", fontsize=16, color='C3',
				horizontalalignment='center',  verticalalignment='center', transform=ax.transAxes)
			col='C3'

		varnames = np.array([
			r'$M_*$', 
			r'SFR',  r'(1+$\delta$)',  r'$M_H$', 'R', r'$\sigma_{c}[M_{BH}]$', r'$\phi_g$'
			])
		varnames=varnames[idx]

		ax.bar(loc, performance, yerr=errors, width=0.6, 
		edgecolor=col, facecolor='w',
		hatch=2*'\\', lw=2) #label=' \n MSE train,test = [{}, {}]'.format(mse_tr.round(3),mse_va.round(3)))
		ax.set_xticks(loc)
		ax.tick_params(which='both', axis='x', bottom=False, top=False)
		ax.set_ylim(0,0.7)
		# varnames = np.array([
		# 	r'$M_*$', 
		# 	r'SFR',  r'(1-$\delta$)',  r'$M_H$', 'R', r'$M_{csig}(\sigma_c)$', r'$\phi_g$'
		# 	]) 
		ax.set_xticklabels(varnames, fontsize=12)
		ax.set_ylabel('Relative importance')
		#ax.legend()

		fig.savefig('Plots/Z_total_performance_combined_{}{}.pdf'.format(self.type_,run), bbox_inches='tight')


	def csigmzr_new_method(self):

		'''plot central velocity dispersion vs stellar metallicity binned in tracks of stellar mass'''

		configure_plots()
		Nsize=25
		fig, ax=plt.subplots(nrows=1, ncols=1, figsize=(10,8))
		Mstar, SFR, ov, hma, ran, Z, csig, phi=self.dat

		# ax.set_xlabel(r'$\mathrm{log(M_{csig}(\sigma_c)/M_\odot)}$', fontsize=Nsize)
		ax.set_xlabel(r'$\mathrm{log(\sigma_c/kms^{-1})}$', fontsize=Nsize)
		ax.set_ylabel(r'$\mathrm{log(Z/Z_\odot)}$', fontsize=Nsize)

		ax.grid()

		bin_means, bin_edges, binnumber=binned_statistic(csig, [csig,Z], statistic='median', bins=10)
		ax.plot(bin_means[0], bin_means[1], label=r'Combined', color='black', linestyle='dotted', lw=3)
		st, q1, q2=bin_std_new(Z, bin_means[1], binnumber)
		#ax.fill_between(bin_means[0], q1, q2, color='black', alpha=.2, linewidth=0)


		n = 4
		#colors = plt.cm.viridis(np.linspace(0.2,1,n))
		if self.type_=='SF':
			#colors = plt.cm.Blues(np.linspace(0.5,1,n))
			colors = plt.cm.PuBu(np.linspace(0.5,1,n))
			colors = plt.cm.viridis(np.linspace(0.0,1,n))
			ax.set_ylim(-0.35,0.25)
			ax.set_title( r'Star-forming'+" galaxies", fontsize=30 , pad=10, color='C10')
		else:
			colors = plt.cm.Reds(np.linspace(0.5,1,n))
			colors = plt.cm.viridis(np.linspace(0.0,1,n))
			ax.set_ylim(-0.09,0.32)
			ax.set_title( r'Passive'+" galaxies", fontsize=30 , pad=10, color='C3')


		idx=np.where((Mstar<np.quantile(Mstar,0.25)))[0]
		m1, b1, z1=Mstar[idx], csig[idx], Z[idx]
		bin_means, bin_edges, binnumber=binned_statistic(b1, [b1,z1], statistic='median', bins=5)
		st, q1, q2=bin_std(z1, bin_means[1], binnumber)
		ax.plot(bin_means[0], bin_means[1], label=r'0%-25% in log($M_*$/$M_\odot$)', color=colors[0], linestyle='dashed')
		ax.fill_between(bin_means[0], q1, q2, color=colors[0], alpha=.2, linewidth=0)

		idx=np.where((Mstar>np.quantile(Mstar,0.25)) & (Mstar<np.quantile(Mstar,0.5)))[0]
		m2, b2, z2=Mstar[idx], csig[idx], Z[idx]
		bin_means, bin_edges, binnumber=binned_statistic(b2, [b2,z2], statistic='median', bins=5)
		st, q1, q2=bin_std(z2, bin_means[1], binnumber)
		ax.plot(bin_means[0], bin_means[1], label=r'25%-50% in log($M_*$/$M_\odot$)', color=colors[1], linestyle='dashed')
		ax.fill_between(bin_means[0], q1, q2, color=colors[1], alpha=.2, linewidth=0)

		idx=np.where((Mstar>np.quantile(Mstar,0.5)) & (Mstar<np.quantile(Mstar,0.75)))[0]
		m3, b3, z3=Mstar[idx], csig[idx], Z[idx]
		bin_means, bin_edges, binnumber=binned_statistic(b3, [b3,z3], statistic='median', bins=5)
		st, q1, q2=bin_std(z3, bin_means[1], binnumber)
		ax.plot(bin_means[0], bin_means[1], label=r'50%-75% in log($M_*$/$M_\odot$)', color=colors[2], linestyle='dashed')
		ax.fill_between(bin_means[0], q1, q2, color=colors[2], alpha=.2, linewidth=0)

		idx=np.where((Mstar>np.quantile(Mstar,0.75)))[0]
		m4, b4, z4=Mstar[idx], csig[idx], Z[idx]
		bin_means, bin_edges, binnumber=binned_statistic(b4, [b4,z4], statistic='median', bins=5)
		st, q1, q2=bin_std(z4, bin_means[1], binnumber)
		ax.plot(bin_means[0], bin_means[1], label=r'75%-100% in log($M_*$/$M_\odot$)', color=colors[3], linestyle='dashed')
		ax.fill_between(bin_means[0], q1, q2, color=colors[3], alpha=.2, linewidth=0)

		ax.legend(fontsize=16)

		#5.246 * np.log10(csig) - 3.77
		f = lambda x: 5.246 * x - 3.77 
		g = lambda x: (x+3.77)/5.246

		print("doing secondary axis")
		ax2 = ax.secondary_xaxis("top", functions=(f,g), xlabel=r'log($M_{BH}/M_\odot$)')
		ax2.set_xlabel(r'log($M_{BH}/M_\odot$)', fontsize=25, labelpad=6)
		ax.tick_params(top=False, bottom=True, left=True, right=True,
		            labelleft=True, labelbottom=True)


		fig.savefig('Plots/Z_csig_{}.pdf'.format(self.type_), bbox_inches='tight')



	def csigmzr_other_new_method(self):

		'''Plots tracks of central velocity dispersion for stellar mass vs stellar metallicity'''

		configure_plots()
		Nsize=25
		fig, ax=plt.subplots(nrows=1, ncols=1, figsize=(9,7))
		Mstar, SFR, ov, hma, ran, Z, csig, phi=self.dat

		#ax.set_xlabel('log($M_*$[$M_\odot$])', fontsize=Nsize)
		ax.set_xlabel(r'$\mathrm{log(M_{*}/M_\odot)}$', fontsize=Nsize)
		ax.set_ylabel(r'$\mathrm{log(Z/Z_\odot)}$', fontsize=Nsize)

		ax.grid()

		bin_means, bin_edges, binnumber=binned_statistic(Mstar, [Mstar,Z], statistic='median', bins=10)
		ax.plot(bin_means[0], bin_means[1], label=r'Combined', color='black', linestyle='dotted', lw=3)
		st, q1, q2=bin_std_new(Z, bin_means[1], binnumber)

		n = 4
		colors = plt.cm.viridis(np.linspace(0,1,n))

		if self.type_=='SF':
			#colors = plt.cm.Blues(np.linspace(0.5,1,n))
			#colors = plt.cm.PuBu(np.linspace(0.5,1,n))
			# colors = plt.cm.viridis(np.linspace(0.5,1,n))
			colors = plt.cm.viridis(np.linspace(0.0,1,n))
			ax.set_ylim(-0.4,0.15)
		else:
			#colors = plt.cm.Reds(np.linspace(0.5,1,n))
			# colors = plt.cm.viridis(np.linspace(0.5,1,n))
			colors = plt.cm.viridis(np.linspace(0.0,1,n))
			ax.set_ylim(-0.11,0.24)


		#idx=np.where((csig>5) & (csig<6))[0]


		idx=np.where((csig<np.quantile(csig,0.25)))[0]
		m1, b1, z1=Mstar[idx], csig[idx], Z[idx]
		bin_means, bin_edges, binnumber=binned_statistic(m1, [m1,z1], statistic='median', bins=5)
		st, q1, q2=bin_std(z1, bin_means[1], binnumber)
		ax.plot(bin_means[0], bin_means[1], label=r'0%-25% in log($\sigma_c/kms^{-1}$)[$M_{BH}$]', color=colors[0], linestyle='dashed')
		ax.fill_between(bin_means[0], q1, q2, color=colors[0], alpha=.2, linewidth=0)

		idx=np.where((csig>np.quantile(csig,0.25)) & (csig<np.quantile(csig,0.5)))[0]
		m2, b2, z2=Mstar[idx], csig[idx], Z[idx]
		bin_means, bin_edges, binnumber=binned_statistic(m2, [m2,z2], statistic='median', bins=5)
		st, q1, q2=bin_std(z2, bin_means[1], binnumber)
		ax.plot(bin_means[0], bin_means[1], label=r'25%-50% in log($\sigma_c/kms^{-1}$)[$M_{BH}$]', color=colors[1], linestyle='dashed')
		ax.fill_between(bin_means[0], q1, q2, color=colors[1], alpha=.2, linewidth=0)

		idx=np.where((csig>np.quantile(csig,0.50)) & (csig<np.quantile(csig,0.75)))[0]
		m3, b3, z3=Mstar[idx], csig[idx], Z[idx]
		bin_means, bin_edges, binnumber=binned_statistic(m3, [m3,z3], statistic='median', bins=5)
		st, q1, q2=bin_std(z3, bin_means[1], binnumber)
		ax.plot(bin_means[0], bin_means[1], label=r'50%-75% in log($\sigma_c/kms^{-1}$)[$M_{BH}$]', color=colors[2], linestyle='dashed')
		ax.fill_between(bin_means[0], q1, q2, color=colors[2], alpha=.2, linewidth=0)

		
		idx=np.where((csig>np.quantile(csig,0.75)))[0]
		m4, b4, z4=Mstar[idx], csig[idx], Z[idx]
		bin_means, bin_edges, binnumber=binned_statistic(m4, [m4,z4], statistic='median', bins=3)
		st, q1, q2=bin_std(z4, bin_means[1], binnumber)
		ax.plot(bin_means[0], bin_means[1], label=r'75%-100% in log($\sigma_c/kms^{-1}$)[$M_{BH}$]', color=colors[3], linestyle='dashed')
		ax.fill_between(bin_means[0], q1, q2, color=colors[3], alpha=.2, linewidth=0)

		ax.legend(fontsize=16)


		fig.savefig('Plots/Z_mstar_{}.pdf'.format(self.type_), bbox_inches='tight')


	


	def masses_combined(self):

		'''2D hexagonal bin plot for stellar mass vs central velocity dispersion colour coded by metallicity'''
		configure_plots()

		Mstar, SFR, ov, hma, ran, Z, csig, phi=self.dat
		Nsize=25

		fig, ax=plt.subplots(nrows=1, ncols=1, figsize=(9,8))

		if self.type_=='SF':
			#colors = 'Blues'
			#colors = 'PuBu'
			colors = 'Spectral_r'
			x1_,y1_=9.0,2.2
			a_err=2.3 #original bootstrapped value
			ax.set_title( r'Star-forming'+" galaxies", fontsize=30 , pad=70, color='C10')
		else:
			#colors = 'Reds'
			#colors = 'PuRd'
			colors='Spectral_r'
			#x1_,y1_=9.8,8.0
			x1_,y1_=9.0,2.35
			a_err=1.5 #original bootstrapped value
			ax.set_title( r'Passive'+" galaxies", fontsize=30, pad=70, color='C3')

		ax.set_ylabel(r'$\mathrm{log(\sigma_c/kms^{-1})[M_{BH}]}$', fontsize=Nsize)

		pl1=ax.hexbin(Mstar,csig, C=Z, cmap=colors, gridsize=40, mincnt=7, reduce_C_function=np.median)
		cent=pl1.get_offsets()
		val=pl1.get_array()
		ax.set_xlabel(r'$\mathrm{log(M_*[M_\odot])}$', fontsize=Nsize)
		#cbar1=fig.colorbar(pl1)
		# cbar1.set_label(r'log($Z/Z_\odot$)', fontsize=Nsize)
		sns.kdeplot(x=Mstar,y=csig,ax=ax,thresh=.10, color='grey', levels=5)
		#ax.grid()
		angles=arrow_error_new(cent[:,0], cent[:,1], val)
		#arrow angle error obtained is given by angles.angle_error.round(1) but I fix above to accepted version

		ax.text(0.2, 0.90, 'Arrow Angle \n'r'$\theta$='+str((90-angles.angle).round(1))+
			 r'$\pm$'+str(a_err)+r'$^{\circ}$',  fontsize=20,
				horizontalalignment='center',  verticalalignment='center', transform=ax.transAxes)
		dx,dy, x1,y1=angles.dim()
		ax.quiver(x1_,y1_, dx, dy, 1, angles=(angles.angle_rad),  scale=5)

		# Create a secondary x-axis at the top
		ax.secondary_xaxis('top')
		# Create space for the colorbar above the plot
		from mpl_toolkits.axes_grid1 import make_axes_locatable

		divider = make_axes_locatable(ax)
		cax = divider.append_axes("top", size="5%", pad=0.0)
		# Add the colorbar to the newly created axes
		cbar = fig.colorbar(pl1, cax=cax, orientation="horizontal")
		cbar.set_label(r'$\mathrm{log(Z/Z_\odot)}$', fontsize=Nsize)
		# Adjust colorbar ticks to face upwards
		cax.xaxis.set_ticks_position('top')
		cax.xaxis.set_label_position('top')


		#5.246 * np.log10(csig) - 3.77
		f = lambda x: 5.246 * x - 3.77 
		g = lambda x: (x+3.77)/5.246

		print("doing secondary axis")
		ax2 = ax.secondary_yaxis(1, functions=(f,g), ylabel=r'$\mathrm{log($M_{BH}/M_\odot)}$')
		ax2.set_ylabel(r'$\mathrm{log(M_{BH}/M_\odot)}$', fontsize=25, labelpad=6)
		ax.tick_params(top=False, bottom=True, left=True, right=False,
		            labelleft=True, labelbottom=True)

		fig.savefig('Plots/mass_comp_{}.pdf'.format(self.type_), bbox_inches='tight')



	def mzr_best_fit(self):

		'''Plot a MZR and fit it'''

		configure_plots()
		Nsize=25
		fig, ax=plt.subplots(nrows=1, ncols=1, figsize=(9,7))
		Mstar, SFR, ov, hma, ran, Z, csig, phi=self.dat

		ax.set_xlabel(r'$\mathrm{log(M_{*}/M_\odot)}$', fontsize=Nsize)
		ax.set_ylabel(r'$\mathrm{log(Z/Z_\odot)}$', fontsize=Nsize)

		ax.grid()

		f, xx, yy=cont(Mstar, Z, bw='scott')
		ax.contourf(xx, yy, f, cmap='binary', levels=[0.3, 0.7, 6.])
		ax.contour(xx,yy,f, levels=[0.3, 0.7, 6.], colors='black')

		(b1,c1),(b1err,c1err),std=odrfunc_new(Mstar, Z)


		C=ax.plot(Mstar,b1*(Mstar-8.0)+c1,
					  label='ODR - 2 parameters'+'\n '+str(b1.round(3))+r'$\pm$('+str(b1err.round(3))
					  +r') log($\sigma_c/kms^{-1}$)'+'\n +'+str(c1.round(3))+
					  r' $\pm$('+str(c1err.round(3))+')\n '+r' $\sigma$ = '+str(std.round(2)), color='black')
		#ax.fill_between()

		m,c =initial_fitly(Mstar, Z)

		ax.legend(fontsize=16)


		fig.savefig('Plots/Z_m_best_fit_{}.pdf'.format(self.type_), bbox_inches='tight')

	# def csigmzr_best_fit_updated(self):

	# 	'''Plot central velocity dispersion vs stellar metallicity then fit it - old version
	# 	No longer needed!
	# 	'''

	# 	configure_plots()
	# 	Nsize=25
	# 	fig, ax=plt.subplots(nrows=1, ncols=1, figsize=(9,7))
	# 	Mstar, SFR, ov, hma, ran, Z, csig, phi=self.dat

	# 	#ax.set_xlabel(r'$\mathrm{log(M_{csig}(\sigma_c)/M_\odot)}$', fontsize=Nsize)
	# 	ax.set_xlabel(r'$\mathrm{log(\sigma_c/kms^{-1})}$', fontsize=Nsize)
	# 	#ax.set_xlabel(r'$\mathrm{\sigma_c \;kms^{-1}}$', fontsize=Nsize)

	# 	ax.set_ylabel(r'$\mathrm{log(Z/Z_\odot)}$', fontsize=Nsize)

	# 	ax.grid()

	# 	sns.kdeplot(x=csig,y=Z,ax=ax,thresh=.10, cmap='binary', levels=5, fill=True, bw_method=0.3)


	# 	(b1,c1),(b1err,c1err),std=odrfunc_new(csig, Z)

	# 	ax.set_ylim(-0.295,0.34)
	# 	ax.set_xlim(1.82,2.8)


	# 	bin_means, bin_edges, binnumber=binned_statistic(csig, 
	# 		[csig, Z], statistic='median', bins=7)
	# 	bin_low, bin_edges, binnumber=binned_statistic(csig, 
	# 		[csig, Z], statistic=lambda Z: np.percentile(Z, 16, axis=None), bins=7)
	# 	bin_high, bin_edges, binnumber=binned_statistic(csig, 
	# 		[csig, Z], statistic=lambda Z: np.percentile(Z, 84, axis=None), bins=7)
	# 	bin_std, bin_edges, binnumber=binned_statistic(csig, 
	# 		[csig, Z], statistic='std', bins=7)

	# 	errs=np.array([bin_means[1]-bin_low[1], bin_high[1]-bin_means[1]])
	# 	ax.errorbar(bin_means[0], bin_means[1], yerr=errs, color='black', marker='s', ms=6)

	# 	m,c1 =error_fitly(bin_means[0], bin_means[1], bin_std[1])

	# 	C=ax.plot(csig,m*(csig-8.0)+c1,
	# 				  label='Best-fit'+'\n '+str(m.round(3))
	# 				  +r'$\mathrm{[log(M_{csig}(\sigma_c))\,-8\,M_{\odot}]}+$'+str(c1.round(3)), color='red', zorder=3)

	

	# 	ax.legend(fontsize=16, loc='lower right')

	# 	f = lambda x: 5.246 * x - 3.77 
	# 	g = lambda x: (x+3.77)/5.246

	# 	print("doing secondary axis")
	# 	ax2 = ax.secondary_xaxis("top", functions=(f,g), xlabel=r'log($M_{BH}/M_\odot$)')
	# 	ax2.set_xlabel(r'log($M_{BH}/M_\odot$)', fontsize=25, labelpad=6)
	# 	ax.tick_params(top=False, bottom=True, left=True, right=False,
	# 	            labelleft=True, labelbottom=True)


	# 	fig.savefig('Plots/Z_csig_best_fit_{}.pdf'.format(self.type_), bbox_inches='tight', dpi=300)



	def sig_best_fit_updated(self):

		'''Plot central velocity dispersion vs stellar metallicity then fit it - new version'''

		configure_plots()
		Nsize=25
		fig, ax=plt.subplots(nrows=1, ncols=1, figsize=(9,7))
		Mstar, SFR, ov, hma, ran, Z, csig, phi=self.dat

		#ax.set_xlabel(r'$\mathrm{log(M_{csig}(\sigma_c)/M_\odot)}$', fontsize=Nsize)
		ax.set_xlabel(r'$\mathrm{log(\sigma_c/kms^{-1})}$', fontsize=Nsize)

		ax.set_ylabel(r'$\mathrm{log(Z/Z_\odot)}$', fontsize=Nsize)

		ax.grid()

		sns.kdeplot(x=csig,y=Z,ax=ax,thresh=.10, cmap='binary', levels=5, fill=True, bw_method=0.3)


		ax.set_ylim(-0.295,0.39)
		ax.set_xlim(1.82,2.65)


		bin_means, bin_edges, binnumber=binned_statistic(csig, 
			[csig, Z], statistic='median', bins=7)
		bin_low, bin_edges, binnumber=binned_statistic(csig, 
			[csig, Z], statistic=lambda Z: np.percentile(Z, 16, axis=None), bins=7)
		bin_high, bin_edges, binnumber=binned_statistic(csig, 
			[csig, Z], statistic=lambda Z: np.percentile(Z, 84, axis=None), bins=7)
		bin_std, bin_edges, binnumber=binned_statistic(csig, 
			[csig, Z], statistic='std', bins=7)

		errs=np.array([bin_means[1]-bin_low[1], bin_high[1]-bin_means[1]])
		ax.errorbar(bin_means[0], bin_means[1], yerr=errs, color='black', marker='s', ms=6)

		# m,c1 =error_fitly(bin_means[0], bin_means[1], bin_std[1])
		(b1,c1),(b1err,c1err),std=odrfunc_new(bin_means[0], bin_means[1])

		(b11,c11),(b11err,c11err),std=odrfunc_new((5.246*bin_means[0]-3.77), bin_means[1])
		# C=ax.plot(csig,m*csig + c1,
		# 			  label='Best-fit'+'\n '+str(m.round(3))
		# 			  +r'$[log($\sigma_c/kms^{-1}$)]+$'+str(c1.round(3)), color='red', zorder=3)
		print(b1.round(3), b1err.round(3), c1.round(3), c1err.round(3))
		C=ax.plot(csig,b1*csig + c1,
					  label='Best-fit'+'\n '+str(b1.round(3))
					  +r'$\;\rm log(\sigma_c/kms^{-1})\;$'+str(c1.round(3))
					  +'\n '+ str(b11.round(3)) + r'$\;\rm log(M_{BH}/M_\odot)\;$'+str(c11.round(3)), color='red', zorder=3)

					  #color='red', zorder=3)

		print(b11.round(3), b11err.round(3), c11.round(3), c11err.round(3))

		ax.legend(fontsize=16, loc='lower right')
		f = lambda x: 5.246 * x - 3.77 
		g = lambda x: (x+3.77)/5.246

		print("doing secondary axis")
		ax2 = ax.secondary_xaxis("top", functions=(f,g), xlabel=r'log($M_{BH}/M_\odot$)')
		ax2.set_xlabel(r'log($M_{BH}/M_\odot$)', fontsize=25, labelpad=6)
		ax.tick_params(top=False, bottom=True, left=True, right=False,
		            labelleft=True, labelbottom=True)


		# fig.savefig('Plots/Z_csig_best_fit_{}.pdf'.format(self.type_), bbox_inches='tight', dpi=300)
		fig.savefig('Plots/Z_csig_best_fit_{}.EPS'.format(self.type_), bbox_inches='tight', dpi=300)

	def save_figure_data_hdf5(self, output_file):
	    """
	    Save all the data used for generating figures to an HDF5 file.
	    
	    Parameters:
	    - output_file (str): File path where the HDF5 file will be saved.
	    """
	    import h5py
	    import numpy as np
	    
	    with h5py.File(output_file, 'w') as hf:
	        # Save centrals data
	        centrals_group = hf.create_group('centrals')
	        for i, name in enumerate(['Mstar', 'SFR', 'OH', 'Overdensity', 'Halo_Mass', 'Random', 'Z', 'csig', 'phi']):
	            centrals_group.create_dataset(name, data=self.dataC[i])
	        
	        # Save satellites data
	        satellites_group = hf.create_group('satellites')
	        for i, name in enumerate(['Mstar', 'SFR', 'OH', 'Overdensity', 'Halo_Mass', 'Random', 'Z', 'csig', 'phi']):
	            satellites_group.create_dataset(name, data=self.dataS[i])
	        
	        # Save Random Forest results
	        rf_results = self.RF_separate()
	        rf_group = hf.create_group('random_forest')
	        rf_group.create_dataset('varnames', data=np.array(rf_results['varnames'], dtype='S'))
	        rf_group.create_dataset('importance_centrals', data=rf_results['performance_C'])
	        rf_group.create_dataset('importance_satellites', data=rf_results['performance_S'])
	        rf_group.create_dataset('errors_centrals', data=rf_results['errors_C'])
	        rf_group.create_dataset('errors_satellites', data=rf_results['errors_S'])
	    
	    print(f"Data saved successfully to {output_file}")


	

	
def track_binning(x,y,z, num_bins=10):
	bin_means, bin_edges, binnumber=binned_statistic(x, [x,y,z], statistic='median', bins=num_bins)
	mass=bin_means[0]
	#lower bin

	upps, lows=[],[]
	mass_l=[]

	#upper bin
	upps_u, lows_u=[], []
	mass_u=[]

	for n in list(set(binnumber)):  # Iterate over each bin
		idx=np.where(binnumber==n)[0]
		in_bin_z=z[idx]
		in_bin_y=y[idx]
		in_bin_x=x[idx]
		index=np.argsort(in_bin_z)
		sorted_y=in_bin_y[index]
		sorted_z=in_bin_z[index]
		sorted_x=in_bin_x[index]
		ind=int((len(sorted_z)-1)/2)
		print(ind)
		ys=sorted_y[:ind]
		#upps.append(max(ys))
		#lows.append(min(ys))
		upps.append(np.quantile(ys,0.50))
		lows.append(np.quantile(ys,0.01))
		mass_l.append(np.median(sorted_x))


		y2=sorted_y[ind:]             
		#upps_u.append(max(ys))
		#lows_u.append(min(ys))
		upps_u.append(np.quantile(y2,0.99))
		lows_u.append(np.quantile(y2,0.50))
		mass_u.append(np.median(sorted_x))


	return mass, bin_means[1], upps, lows, upps_u, lows_u, mass_u, mass_u











def bin_std_new(quantity_pre_bin, quantity_post_bin, binnumber, ran=False):
	stds = []
	st2=[]
	qup=[]
	qlow=[]
	# Calculate stdev for all elements inside each bin

	for n in list(set(binnumber)):  # Iterate over each bin
		idx=np.where(binnumber==n)[0]
		in_bin=quantity_pre_bin[idx]
		if ran==False:
			stds.append(np.std(in_bin)/np.sqrt(len(in_bin)))
			st2.append(np.std(in_bin)/np.sqrt(len(in_bin)))
			qup.append(np.quantile(in_bin, 0.84))
			qlow.append(np.quantile(in_bin, 0.16))
		else:
			if n!=min(set(binnumber)) and n!=max(set(binnumber)):
				stds.append(np.std(in_bin)/np.sqrt(len(in_bin)))
				st2.append(np.std(in_bin)/np.sqrt(len(in_bin)))
		#print(len(in_bin))
	q1=quantity_post_bin - qlow
	q2=qup - quantity_post_bin 
	return stds, qlow, qup


class manga_1(object):

	def __init__(self, star_forming=True):

		self.star_forming=star_forming

		if star_forming:
			self.type_='SF'
		else:
			self.type_='Q'

		super(manga_1,self).__init__()
		data=fits.open('data/manga_4may_stellarZ.fits')[1].data

		Z=data['Z']
		Mdyn=data['logdyn']
		sigma_e=data['logsigma_e']
		age=data['logAge']
		Morph=data['Morphology']
		sfr=data['log_SFR_Ha_1']
		Mstar=data['log_Mass_1']
		r_e=data['logr_e'] #Cappelarri r-e
		r_e=data['Re_kpc_1'] #Sanchez_r_e
		vel_ssp=data['sigma_cen'] #sanchez
		Z=data['zh_mw_re_fit_1']


		ms_sfr=ms_randp(Mstar)
		passive=sfr-ms_sfr<-0.75
		sf=sfr-ms_sfr>-0.75

		if star_forming==True:
			state=sf
		else:
			state=passive

		
		# print(Z.shape)
	
		# print(np.mean(vel_ssp), np.mean(Mdyn), np.mean(sfr), np.mean(Mstar), np.mean(Z))
		idx=np.where((Mstar > 1) & (Mstar<1000) 
			 & (sfr>-12) & (sfr<10) & (state) & (vel_ssp>0) & (vel_ssp<10000) & (Z>-20) & (Z<300))[0]
		# idx=np.where( (Mdyn>1) & (Mdyn<100) & (Mstar > 1) 
		# 	 )[0]
		#print(idx)
		Mstar=Mstar[idx]
		Mdyn, Z=Mdyn[idx], Z[idx]
		sfr, vel_sig=sfr[idx], vel_ssp[idx]
		r_e=r_e[idx]
		print(len(Mstar))
		#print(np.mean(vel_ssp), np.mean(Mdyn), np.mean(sfr), np.mean(Mstar), np.mean(sigma_e))

		#print(vel_ssp, Mdyn, sfr, Mstar, sigma_e)
		

		self.dat=self.dataC=OC(np.array((Mstar,  Z, 
			sfr, vel_sig, Mdyn, r_e)))



	def RF(self):

		configure_plots()

		[Mstar, Z, SFR, vel_sig, Mdyn, r_e]=self.dat

		output_dir=f'/Users/will/GitHub/Stellar_Z_project/data/'
		file_name = os.path.join(output_dir, f"RF_output_summary_manga_{self.type_}_1.pkl")

		if os.path.isfile(file_name):
			with open(file_name, "rb") as input_file:
				output_summary = pickle.load(input_file)
			eout=output_summary
			#print(eout)
			performance=eout['performance_sorted']
			errors=eout['errors_sorted']
			loc=eout['loc']
			varnames=eout['varnames']
			mse_test=eout['mse_test']
			mse_train=eout['mse_train']
			print(f"MSE_test: {mse_test}  MSE_train: {mse_train}")
		else:

			[Mstar, Z, SFR, vel_sig, Mdyn, r_e]=self.dat
			#csig = 5.246 * np.log10(vel_sig) - 3.77
			csig = np.log10(vel_sig)
			# csig=sigma_e
			# csig=vel_sig
			phi=Mdyn-r_e
			#MstarC, SFRC=MstarC+np.random.normal(0,0.2,len(MstarC)), SFRC+np.random.normal(0,0.2,len(MstarC))
			ran=np.random.uniform(0,1.,len(Z))
			print(len(Mstar))
			data=np.array([Z, Mstar, SFR, ran, csig, phi, Mdyn])

			#csig = 5.246 * np.log10(csig) - 3.77
			performance, mse_tr, mse_va, length=basic_RF(data, n_est=200, min_samp_leaf=15, max_dep=100, param_check=False) 
			performance, errors=boots_RFR_updated(data,  n_est=200, min_samp_leaf=15, max_dep=100, n_times=100, wide=False)


			varnames = np.array([
			r'$M_*$',
			r'SFR', 'Random', r'$\sigma_c$', r'$\phi_g$', r'$M_{Dyn}$'
			])
			# varnames = np.array([
			# r'$M_*$',
			# r'SFR', r'$M_{Dyn}$', 'Random', r'$\sigma_{e}$', r'$\phi$'
			# ])

			idx = np.argsort(performance)[::-1]

			loc = np.arange(length.shape[1])

			performance=performance[idx]
			errors=errors[:,idx]
			varnames=varnames[idx]

			RF_data={}
			RF_data['performance_sorted']=performance
			RF_data['errors_sorted']=errors
			RF_data['loc']=loc
			RF_data['mse_train']=mse_tr.round(3)
			RF_data['mse_test']=mse_va.round(3)
			RF_data['varnames']=varnames
			output={}
			output['results']=RF_data
			output_dir=f'/Users/will/GitHub/Stellar_Z_project/data/'
			if os.path.exists(file_name):
				os.remove(file_name)
			f = open(file_name, "wb")
			pickle.dump(RF_data, f, protocol=pickle.HIGHEST_PROTOCOL)

		fig, ax = plt.subplots(figsize=(5,5))
		if self.star_forming:
			# ax.text(0.6, 0.7, "Parameter importance \n in determining Z for \n"+ r'star-forming' +" galaxies", fontsize=16,
			# 	horizontalalignment='center',  verticalalignment='center', transform=ax.transAxes)
			# ax.text(0.6, 0.7, "Parameter importance \n in determining the \n metallicity \n Star Forming", fontsize=16,
			# 	horizontalalignment='center',  verticalalignment='center', transform=ax.transAxes)
			ax.text(0.6, 0.8, "Parameter importance \n in determining the \n metallicity \n \n", fontsize=16,
				horizontalalignment='center',  verticalalignment='center', transform=ax.transAxes)
			ax.text(0.6, 0.7, r"Star Forming", fontsize=16, color='C10',
				horizontalalignment='center',  verticalalignment='center', transform=ax.transAxes)
			col='#0F2080'
			col='C10'
		else:
			# ax.text(0.6, 0.7, "Parameter importance \n in determining Z for \n passive galaxies", fontsize=16,
			# 	horizontalalignment='center',  verticalalignment='center', transform=ax.transAxes)
			# ax.text(0.6, 0.8, "Parameter importance \n in determining the \n metallicity \n \n"+r"$\bf Passive$", fontsize=16,
			# 	horizontalalignment='center',  verticalalignment='center', transform=ax.transAxes)
			ax.text(0.6, 0.8, "Parameter importance \n in determining the \n metallicity \n \n", fontsize=16,
				horizontalalignment='center',  verticalalignment='center', transform=ax.transAxes)
			ax.text(0.6, 0.7, r"Passive", fontsize=16, color='C3',
				horizontalalignment='center',  verticalalignment='center', transform=ax.transAxes)
			col='C3'

		ax.bar(loc, performance, yerr=errors, width=0.6, 
		edgecolor=col, facecolor='w',
		hatch=2*'\\', lw=2) #label=' \n MSE train,test = [{}, {}]'.format(mse_tr.round(3),mse_va.round(3)))
		ax.set_xticks(loc)
		ax.tick_params(which='both', axis='x', bottom=False, top=False)
		ax.set_ylim(0,0.7)
		ax.set_xticklabels(varnames, fontsize=13)
		ax.set_ylabel('Relative importance')
		ax.set_title(f"MaNGA {len(Mstar)} galaxies")
		#ax.legend()
		
		fig.savefig('Plots/Z_total_performance_MANGA_{}_1.pdf'.format(self.type_), bbox_inches='tight')

class manga_2(object):

	def __init__(self, star_forming=True):

		self.star_forming=star_forming

		if star_forming:
			self.type_='SF'
		else:
			self.type_='Q'

		super(manga_2,self).__init__()
	

		data=fits.open('/Users/will/GitHub/Stellar_Z_project/data/pipe_brownson_12jul.fits')[1].data
		# Z=data['zh_lw_re_fit']
		Z=data['zh_mw_re_fit']
		Mdyn=data['Dynamical Mass (Inclined Rotating Disc Model)']
		phi_g=np.array(data['<V> (Inclined Rotating Disc Model)'])**2 + 3 * np.array(data['<Sigma> (Inclined Rotating Disc Model)'])**2
		#sigma_e=data['logsigma']
		# age=data['logAge']
		# Morph=data['Morphology']
		sfr=data['log_sfr_ha']
		Mstar=data['log_Mass']
		# r_e=data['logr_e'] #Cappelarri r-e
		# r_e=data['Re_kpc_1'] #Sanchez_r_e
		vel_ssp=data['sigma_cen'] #sanchez
		# Z=data['zh_mw_re_fit_1']


		ms_sfr=ms_randp(Mstar)
		passive=sfr-ms_sfr<-0.75
		sf=sfr-ms_sfr>-0.75

		if star_forming==True:
			state=sf
		else:
			state=passive

		
		print(Z.shape)
		phi_g=np.log10(phi_g)
		print(np.mean(vel_ssp), np.mean(Mdyn), np.mean(sfr), np.mean(Mstar), np.mean(Z)), np.mean(phi_g)
		print(phi_g)
		idx=np.where((Mstar > 1) & (Mstar<1000) 
			 & (sfr>-12) & (sfr<10) & (state) & (vel_ssp>0) & (vel_ssp<10000) & (Z>-20) & (Z<300) & (phi_g>0.1) & (phi_g<10000000))[0]
		# idx=np.where( (Mdyn>1) & (Mdyn<100) & (Mstar > 1) 
		# 	 )[0]
		#print(idx)
		Mstar=Mstar[idx]
		Mdyn, Z=Mdyn[idx], Z[idx]
		sfr, vel_sig=sfr[idx], vel_ssp[idx]
		phi_g=phi_g[idx]
		print(len(Mstar))
		#print(np.mean(vel_ssp), np.mean(Mdyn), np.mean(sfr), np.mean(Mstar), np.mean(sigma_e))

		#print(vel_ssp, Mdyn, sfr, Mstar, sigma_e)
		

		self.dat=self.dataC=OC(np.array((Mstar,  Z, 
			sfr, vel_sig, phi_g, Mdyn)))



	def RF(self):

		configure_plots()

		[Mstar, Z, SFR, vel_sig, phi_g, Mdyn]=self.dat

		output_dir=f'/Users/will/GitHub/Stellar_Z_project/data/'
		file_name = os.path.join(output_dir, f"RF_output_summary_manga_{self.type_}.pkl")

		if os.path.isfile(file_name):
			with open(file_name, "rb") as input_file:
				output_summary = pickle.load(input_file)
			eout=output_summary
			#print(eout)
			performance=eout['performance_sorted']
			errors=eout['errors_sorted']
			loc=eout['loc']
			varnames=eout['varnames']
			mse_test=eout['mse_test']
			mse_train=eout['mse_train']
			print(f"MSE_test: {mse_test}  MSE_train: {mse_train}")
		else:

			[Mstar, Z, SFR, vel_sig, phi_g, Mdyn]=self.dat
			#csig = 5.246 * np.log10(vel_sig) - 3.77
			csig=np.log10(vel_sig)
			# csig=sigma_e
			# csig=vel_sig
			#phi=Mdyn-r_e
			#MstarC, SFRC=MstarC+np.random.normal(0,0.2,len(MstarC)), SFRC+np.random.normal(0,0.2,len(MstarC))
			ran=np.random.uniform(0,1.,len(Z))
			print(len(Mstar))
			data=np.array([Z, Mstar, SFR, ran, csig, phi_g])

			#csig = 5.246 * np.log10(csig) - 3.77
			performance, mse_tr, mse_va, length=basic_RF(data, n_est=200, min_samp_leaf=15, max_dep=100, param_check=False) 
			performance, errors=boots_RFR_updated(data,  n_est=200, min_samp_leaf=15, max_dep=100, n_times=100, wide=False)


			varnames = np.array([
			r'$M_*$',
			r'SFR', 'Random', r'$\sigma_c$', r'$\phi_{g}$'
			])
			# varnames = np.array([
			# r'$M_*$',
			# r'SFR', r'$M_{Dyn}$', 'Random', r'$\sigma_{e}$', r'$\phi$'
			# ])

			idx = np.argsort(performance)[::-1]

			loc = np.arange(length.shape[1])

			performance=performance[idx]
			errors=errors[:,idx]
			varnames=varnames[idx]

			RF_data={}
			RF_data['performance_sorted']=performance
			RF_data['errors_sorted']=errors
			RF_data['loc']=loc
			RF_data['mse_train']=mse_tr.round(3)
			RF_data['mse_test']=mse_va.round(3)
			RF_data['varnames']=varnames
			output={}
			output['results']=RF_data
			output_dir=f'/Users/will/GitHub/Stellar_Z_project/data/'
			if os.path.exists(file_name):
				os.remove(file_name)
			f = open(file_name, "wb")
			pickle.dump(RF_data, f, protocol=pickle.HIGHEST_PROTOCOL)

		fig, ax = plt.subplots(figsize=(5,5))
		if self.star_forming:
			# ax.text(0.6, 0.7, "Parameter importance \n in determining Z for \n"+ r'star-forming' +" galaxies", fontsize=16,
			# 	horizontalalignment='center',  verticalalignment='center', transform=ax.transAxes)
			# ax.text(0.6, 0.8, "Parameter importance \n in determining the \n metallicity \n \n"+r"$\bf Star\; Forming$", fontsize=16,
			# 	horizontalalignment='center',  verticalalignment='center', transform=ax.transAxes)
			ax.text(0.6, 0.8, "Parameter importance \n in determining the \n metallicity \n \n ", fontsize=16,
				horizontalalignment='center',  verticalalignment='center', transform=ax.transAxes)
			ax.text(0.6, 0.7, r"Star Forming", fontsize=16, color='C10',
				horizontalalignment='center',  verticalalignment='center', transform=ax.transAxes)
			col='#0F2080'
			col='C10'
		else:
			# ax.text(0.6, 0.7, "Parameter importance \n in determining Z for \n passive galaxies", fontsize=16,
			# 	horizontalalignment='center',  verticalalignment='center', transform=ax.transAxes)
			# ax.text(0.6, 0.8, "Parameter importance \n in determining the \n metallicity \n \n"+r"$\bf Passive$", fontsize=16,
			# 	horizontalalignment='center',  verticalalignment='center', transform=ax.transAxes)
			ax.text(0.6, 0.8, "Parameter importance \n in determining the \n metallicity \n \n", fontsize=16,
				horizontalalignment='center',  verticalalignment='center', transform=ax.transAxes)
			ax.text(0.6, 0.7, r"Passive", fontsize=16, color='C3',
				horizontalalignment='center',  verticalalignment='center', transform=ax.transAxes)
			col='C3'

		ax.bar(loc, performance, yerr=errors, width=0.6, 
		edgecolor=col, facecolor='w',
		hatch=2*'\\', lw=2) #label=' \n MSE train,test = [{}, {}]'.format(mse_tr.round(3),mse_va.round(3)))
		ax.set_xticks(loc)
		ax.tick_params(which='both', axis='x', bottom=False, top=False)
		ax.set_ylim(0,0.7)
		ax.set_xticklabels(varnames, fontsize=13)
		ax.set_ylabel('Relative importance')
		ax.set_title(f"MaNGA {len(Mstar)} galaxies")
		#ax.legend()
		
		fig.savefig('Plots/Z_total_performance_MANGA_{}.pdf'.format(self.type_), bbox_inches='tight')


class manga_3(object):

	def __init__(self, star_forming=True):

		self.star_forming=star_forming

		if star_forming:
			self.type_='SF'
		else:
			self.type_='Q'

		super(manga_3,self).__init__()
		

		data=fits.open('/Users/will/GitHub/Stellar_Z_project/data/pipe_brownson_12jul.fits')[1].data
		# Z=data['zh_lw_re_fit']
		Z=data['zh_mw_re_fit']
		Mdyn=data['Dynamical Mass (Inclined Rotating Disc Model)']
		phi_g=np.array(data['<V> (Inclined Rotating Disc Model)'])**2 + 3 * np.array(data['<Sigma> (Inclined Rotating Disc Model)'])**2
		#sigma_e=data['logsigma']
		# age=data['logAge']
		# Morph=data['Morphology']
		sfr=data['log_sfr_ha']
		Mstar=data['log_Mass']
		# r_e=data['logr_e'] #Cappelarri r-e
		# r_e=data['Re_kpc_1'] #Sanchez_r_e
		vel_ssp=data['sigma_cen'] #sanchez
		# Z=data['zh_mw_re_fit_1']


		ms_sfr=ms_randp(Mstar)
		passive=sfr-ms_sfr<-0.75
		sf=sfr-ms_sfr>-0.75

		if star_forming==True:
			state=sf
		else:
			state=passive

		
		print(Z.shape)
		phi_g=np.log10(phi_g)
		print(np.mean(vel_ssp), np.mean(Mdyn), np.mean(sfr), np.mean(Mstar), np.mean(Z)), np.mean(phi_g)
		print(phi_g)
		idx=np.where((Mstar > 1) & (Mstar<1000) 
			 & (sfr>-12) & (sfr<10) & (state) & (vel_ssp>0) & (vel_ssp<10000) & (Z>-20) & (Z<300) & (phi_g>0.1) & (phi_g<10000000))[0]
		# idx=np.where( (Mdyn>1) & (Mdyn<100) & (Mstar > 1) 
		# 	 )[0]
		#print(idx)
		Mstar=Mstar[idx]
		Mdyn, Z=Mdyn[idx], Z[idx]
		sfr, vel_sig=sfr[idx], vel_ssp[idx]
		phi_g=phi_g[idx]
		print(len(Mstar))
		#print(np.mean(vel_ssp), np.mean(Mdyn), np.mean(sfr), np.mean(Mstar), np.mean(sigma_e))

		#print(vel_ssp, Mdyn, sfr, Mstar, sigma_e)
		

		self.dat=self.dataC=OC(np.array((Mstar,  Z, 
			sfr, vel_sig, phi_g, Mdyn)))



	def RF(self):

		configure_plots()

		[Mstar, Z, SFR, vel_sig, phi_g, Mdyn]=self.dat

		output_dir=f'/Users/will/GitHub/Stellar_Z_project/data/'
		file_name = os.path.join(output_dir, f"RF_output_summary_manga_{self.type_}_dyn.pkl")

		if os.path.isfile(file_name):
			with open(file_name, "rb") as input_file:
				output_summary = pickle.load(input_file)
			eout=output_summary
			#print(eout)
			performance=eout['performance_sorted']
			errors=eout['errors_sorted']
			loc=eout['loc']
			varnames=eout['varnames']
			mse_test=eout['mse_test']
			mse_train=eout['mse_train']
			print(f"MSE_test: {mse_test}  MSE_train: {mse_train}")
		else:

			[Mstar, Z, SFR, vel_sig, phi_g, Mdyn]=self.dat
			#csig = 5.246 * np.log10(vel_sig) - 3.77
			csig = np.log10(vel_sig)
			# csig=sigma_e
			# csig=vel_sig
			#phi=Mdyn-r_e
			#MstarC, SFRC=MstarC+np.random.normal(0,0.2,len(MstarC)), SFRC+np.random.normal(0,0.2,len(MstarC))
			ran=np.random.uniform(0,1.,len(Z))
			print(len(Mstar))
			data=np.array([Z, Mstar, SFR, ran, csig, phi_g, Mdyn])

			#csig = 5.246 * np.log10(csig) - 3.77
			performance, mse_tr, mse_va, length=basic_RF(data, n_est=200, min_samp_leaf=15, max_dep=100, param_check=False) 
			performance, errors=boots_RFR_updated(data,  n_est=200, min_samp_leaf=15, max_dep=100, n_times=100, wide=False)


			varnames = np.array([
			r'$M_*$',
			r'SFR', 'Random', r'$\sigma_c$', r'$\phi_{g}$', r'$M_{Dyn}$'
			])
			# varnames = np.array([
			# r'$M_*$',
			# r'SFR', r'$M_{Dyn}$', 'Random', r'$\sigma_{e}$', r'$\phi$'
			# ])

			idx = np.argsort(performance)[::-1]

			loc = np.arange(length.shape[1])

			performance=performance[idx]
			errors=errors[:,idx]
			varnames=varnames[idx]

			RF_data={}
			RF_data['performance_sorted']=performance
			RF_data['errors_sorted']=errors
			RF_data['loc']=loc
			RF_data['mse_train']=mse_tr.round(3)
			RF_data['mse_test']=mse_va.round(3)
			RF_data['varnames']=varnames
			output={}
			output['results']=RF_data
			output_dir=f'/Users/will/OneDrive - University of Cambridge/PythonCoding/Plots/stellar_Z/data/'
			if os.path.exists(file_name):
				os.remove(file_name)
			f = open(file_name, "wb")
			pickle.dump(RF_data, f, protocol=pickle.HIGHEST_PROTOCOL)

		fig, ax = plt.subplots(figsize=(5,5))
		if self.star_forming:
			# ax.text(0.6, 0.7, "Parameter importance \n in determining Z for \n"+ r'star-forming' +" galaxies", fontsize=16,
			# 	horizontalalignment='center',  verticalalignment='center', transform=ax.transAxes)
			# ax.text(0.6, 0.8, "Parameter importance \n in determining the \n metallicity \n \n"+r"$\bf Star\; Forming$", fontsize=16,
			# 	horizontalalignment='center',  verticalalignment='center', transform=ax.transAxes)
			ax.text(0.6, 0.8, "Parameter importance \n in determining the \n metallicity \n \n ", fontsize=16,
				horizontalalignment='center',  verticalalignment='center', transform=ax.transAxes)
			ax.text(0.6, 0.7, r"Star Forming", fontsize=16, color='C10',
				horizontalalignment='center',  verticalalignment='center', transform=ax.transAxes)
			col='#0F2080'
			col='C10'
		else:
			# ax.text(0.6, 0.7, "Parameter importance \n in determining Z for \n passive galaxies", fontsize=16,
			# 	horizontalalignment='center',  verticalalignment='center', transform=ax.transAxes)
			# ax.text(0.6, 0.8, "Parameter importance \n in determining the \n metallicity \n \n"+r"$\bf Passive$", fontsize=16,
			# 	horizontalalignment='center',  verticalalignment='center', transform=ax.transAxes)
			ax.text(0.6, 0.8, "Parameter importance \n in determining the \n metallicity \n \n", fontsize=16,
				horizontalalignment='center',  verticalalignment='center', transform=ax.transAxes)
			ax.text(0.6, 0.7, r"Passive", fontsize=16, color='C3',
				horizontalalignment='center',  verticalalignment='center', transform=ax.transAxes)
			col='C3'

		ax.bar(loc, performance, yerr=errors, width=0.6, 
		edgecolor=col, facecolor='w',
		hatch=2*'\\', lw=2) #label=' \n MSE train,test = [{}, {}]'.format(mse_tr.round(3),mse_va.round(3)))
		ax.set_xticks(loc)
		ax.tick_params(which='both', axis='x', bottom=False, top=False)
		ax.set_ylim(0,0.7)
		ax.set_xticklabels(varnames, fontsize=13)
		ax.set_ylabel('Relative importance')
		ax.set_title(f"MaNGA {len(Mstar)} galaxies")
		#ax.legend()
		
		fig.savefig('Plots/Z_total_performance_MANGA_{}.pdf'.format(self.type_), bbox_inches='tight')


def line(pars, x, y, data=None, err=None):
	'''Simple model of a line for LMFIT'''
	
	vals = pars.valuesdict()

	c=vals['c']
	m=vals['m']

	model= c + m*x
 
	if data is None:
		return model
	
	if err is None:
		return model-data

	return (model - data)/err

def initial_fitly(x,y):
	''' LMFIT fits the line for the sigma ZR'''
	fit_params = Parameters()
	fit_params.add('c', value=0.2, min=-11, max=11)
	fit_params.add('m', value=0.14, min=-3, max=3)
	out_mc = minimize(line, fit_params, args=(x,y), kws={'data': y})
	# out_mc = minimize(line, fit_params, args=(x,y), kws={'data': y}, method='emcee', 
	# 		   nan_policy='omit', burn=100, steps=2000, thin=200,
	# 			  is_weighted=False, progress=True)
	print(fit_report(out_mc))
	#emcee_plot = corner.corner(out_mc.flatchain, labels=out_mc.var_names,
	#				   truths=list(out_mc.params.valuesdict().values()))
	#emcee_plot.savefig('Plots/high_z/corner.pdf')
	m1=np.asarray(out_mc.params['m'].value)
	c1=np.asarray(out_mc.params['c'].value)
	return m1, c1

def error_fitly(x,y, yerr):
	'''LMFIT fits the line taking into account errors'''
	fit_params = Parameters()
	fit_params.add('c', value=0.2, min=-11, max=11)
	fit_params.add('m', value=0.14, min=-3, max=3)
	out_mc = minimize(line, fit_params, args=(x,y), kws={'data': y, 'err': yerr})
	out_mc = minimize(line, fit_params, args=(x,y), kws={'data': y, 'err': yerr}, method='emcee', 
			   nan_policy='omit', burn=100, steps=2000, thin=200,
				  is_weighted=True, progress=True)
	print(fit_report(out_mc))
	#emcee_plot = corner.corner(out_mc.flatchain, labels=out_mc.var_names,
	#				   truths=list(out_mc.params.valuesdict().values()))
	#emcee_plot.savefig('Plots/high_z/corner.pdf')
	m1=np.asarray(out_mc.params['m'].value)
	c1=np.asarray(out_mc.params['c'].value)

	return m1, c1


def odrfunc_new(x,y):
	'''Orthogonal distance regression function with residual standard deviation'''
	linear_func = lambda B, x: B[0]*x +B[1]
	linear=odr.Model(linear_func)
	mydata = odr.RealData(x, y)
	myodr = odr.ODR(mydata, linear, beta0=[1., 2.])

	myoutput = myodr.run()
	#myoutput.pprint()
	r=myoutput.beta
	rerr=myoutput.sd_beta
	re=myoutput.res_var
	
	std=np.sqrt(re)
	return r, rerr, std

def odrfunc_bh(x,y):
	'''Orthogonal distance regression function with residual standard deviation'''
	linear_func = lambda B, x: B[0]*(x-8) +B[1]
	linear=odr.Model(linear_func)
	mydata = odr.RealData(x, y)
	myodr = odr.ODR(mydata, linear, beta0=[1., 2.])

	myoutput = myodr.run()
	#myoutput.pprint()
	r=myoutput.beta
	rerr=myoutput.sd_beta
	re=myoutput.res_var
	
	std=np.sqrt(re)
	return r, rerr, std


def bin_std(quantity_pre_bin, quantity_post_bin, binnumber, ran=False):
	stds = []
	st2=[]
	# Calculate stdev for all elements inside each bin

	for n in list(set(binnumber)):  # Iterate over each bin
		idx=np.where(binnumber==n)[0]
		in_bin=quantity_pre_bin[idx]
		if ran==False:
			stds.append(np.std(in_bin)/np.sqrt(len(in_bin)))
			st2.append(np.std(in_bin)/np.sqrt(len(in_bin)))
		else:
			if n!=min(set(binnumber)) and n!=max(set(binnumber)):
				stds.append(np.std(in_bin)/np.sqrt(len(in_bin)))
				st2.append(np.std(in_bin)/np.sqrt(len(in_bin)))
		#print(len(in_bin))
	q1=quantity_post_bin - st2
	q2=quantity_post_bin + st2
	return stds, q1, q2


if __name__ == "__main__": 
	
	start_time=time.time()

	parser=argparse.ArgumentParser(description='Prospector plotting')
	parser.add_argument( '--sf', help='SF?', type=bool, default=False)
	args = parser.parse_args()

	z_plots=Z_plots_csig(star_forming=args.sf,  SF_definition_new=True)
	z_plots.RF_combined(run='16Apr_sig')
	z_plots.RF_separate()

	z_plots.csigmzr_new_method()
	
	z_plots.csigmzr_other_new_method()

	# if args.star_forming:
	# 		type_='SF'
	# else:
	# 		type_='Q'
	z_plots.save_figure_data_hdf5(f'/Users/will/GitHub/Stellar_Z_project/data/Stellar_Z_paper_{z_plots.type_}_figure_data.h5')

	#z_plots.masses_combined()

	###z_plots.csigmzr_best_fit_updated()

	z_plots.sig_best_fit_updated()

	# # a.stats()


	# m=manga_1(star_forming=args.sf)
	# m.RF()

	# m=manga_2(star_forming=args.sf)
	# m.RF()

	# m=manga_3(star_forming=args.sf)
	# m.RF()

	end_time=time.time()
	print("This took {} seconds!".format(end_time-start_time))
