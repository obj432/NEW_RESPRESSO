from classy import Class # comment this out if you do not have classy installed. You need this to use the mode 3.
import numpy as np
import sys
from scipy.interpolate import InterpolatedUnivariateSpline as ius
import respresso


if __name__ == '__main__':
	args = sys.argv
	recon_mode = int(args[1])
	argc = len(args)
	respresso_obj = respresso.respresso_core()
	if recon_mode == 0:
		print u'reconstruction from linear power spectrum table (normalized at the target redshift)'
		if argc != 4:
			print u'Error! The number of arguments must be 3 for this mode of reconstruction.'
			print u'Correct argument list: recon_mode(=0), linearpowername, outname'
			sys.exit(1)
		plin_data = np.loadtxt(args[2]).T
		plin_spl = ius(plin_data[0,],plin_data[1,])
		outname = args[3]
	elif recon_mode == 1:
		print u'reconstruction from linear power spectrum table (normalized at z=0)'
		if argc != 7:
			print u'Error! The number of arguments must be 6 for this mode of reconstruction.'
			print u'Correct argument list: recon_mode(=1), linearpowername, outname, Om, w, z'
			sys.exit(1)
		Om, w, z = float(args[4]), float(args[5]), float(args[6])
		print u'Provided parameters: Om =', Om, ', w =', w, ' z =', z, '(will be used to compute the linear growth factor)'
		ascale = 1./(1.+z)
		D2 = respresso.linearGrowth(ascale,Om,w)**2
		plin_data = np.loadtxt(args[2]).T
		plin_spl = ius(plin_data[0,],D2*plin_data[1,])
		outname = args[3]
	elif recon_mode == 2:
		print u'reconstruction from transfer table (CAMB normalization at z=0)'
		if argc != 11:
			print u'Error! The number of arguments must be 10 for this mode of reconstruction.'
			print u'Correct argument list: recon_mode(=2), transfername, outname, Om, h0, As, ns, k0(in 1/Mpc), w, z'
			sys.exit(1)
		ks = np.logspace(-3,1,400)
		Om, h0, As, ns, k0, w, z = float(args[4]), float(args[5]), float(args[6]), float(args[7]), float(args[8]), float(args[9]), float(args[10])
		print u'Provided parameters: Om =', Om, ', h0 =', h0, ', As =', As, ', ns =', ns, ', k0 =', k0, ', w =', w, ', z =', z
		cosm = respresso.Cosmology(Om,h0,As,ns,k0,w,args[2]) # Om0, hubble, As, ns, k0[1/Mpc], w and transfer table"
		plin_data = cosm.get_plin(ks,z)
		plin_spl = ius(ks,plin_data)
		outname = args[3]
	elif recon_mode == 3:
		print u'reconstruction from cosmological parameters (linear calculation based on classy, which must be installed to utilize this option)'
		if argc != 11:
			print u'Error! The number of arguments must be 10 for this mode of reconstruction.'
			print u'Correct argument list: recon_mode(=3), outname, Ob, Ocdm, h0, normalization(ln10^10As), ns, k0(in 1/Mpc), w, z'
			sys.exit(1)
		cosmo = Class()
		Ob, Ocdm, h0, normalization, ns, k0, w, z = float(args[3]), float(args[4]), float(args[5]), float(args[6]), float(args[7]), float(args[8]), float(args[9]), float(args[10])
		print u'Provided parameters: Ob =', Ob, ', Ocdm =', Ocdm, ', h0 =', h0, ', ln(10^10As) =', normalization, ', ns =', ns, ', k0 =', k0, ', w =', w, ', z =', z
		if np.fabs(w+1) < 1e-5:
			params = {'output': 'mPk','T_cmb': '2.7255','h': h0,'Omega_b': Ob,'Omega_cdm': Ocdm,'tau_reio': '0.079','n_s': ns,'ln10^{10}A_s': normalization,'Omega_k': '0.0','k_pivot': k0,'P_k_max_h/Mpc': '10.0','z_max_pk': '10'}
		else:
			Ofld = 1. - Ob - Ocdm
			params = {'output': 'mPk','T_cmb': '2.7255','h': h0,'Omega_b': Ob,'Omega_cdm': Ocdm,'Omega_fld': Ofld,'tau_reio': '0.079','n_s': ns,'ln10^{10}A_s': normalization,'w0_fld': w,'Omega_k': '0.0','k_pivot': k0,'P_k_max_h/Mpc': '10.0','z_max_pk': '10'}
		cosmo.set(params)
		cosmo.compute()
		ks = np.logspace(-3,1,400)
		plin_data = [cosmo.pk(ks[i]*cosmo.h(), z)*cosmo.h()**3 for i in range(400)]
		plin_spl = ius(ks,plin_data)
		outname = args[2]
	else:
		print u'Error! reconstruction mode', recon_mode, 'is not supported currently. Please choose an integer from 0, 1, 2, and 3.'
		sys.exit(1)
	respresso_obj.set_target(plin_spl)
	respresso_obj.find_path()
	kmax = respresso_obj.get_kmax()
	print u'Rough estimate of the maximum wavenumber: k_max =', kmax, ' h/Mpc'
	kwave = respresso_obj.get_kinternal()
	pnl_rec = respresso_obj.reconstruct()
	np.savetxt(outname,np.array([kwave,pnl_rec]).T)

