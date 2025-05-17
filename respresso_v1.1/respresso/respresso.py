import os
import numpy as np
from scipy import ndimage
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import integrate
from sklearn.externals import joblib

class respresso_core:	
	def __init__(self):
		print ('Hello. This is RESPRESSO.')
		self.resp = ResponseData()
		self.cfid = FiducialCosmology()
		self.qs = np.linspace(0.0025,1.5025,301)
		self.ks = np.linspace(0.0025,1.0025,201)
		print ('RESPRESSO ready.')

	def set_target(self,plin,t=3,s=0.25,thre=1.):
		self.tilt = t
		self.s_target = s * 2.*np.pi**2
		self.thre_kmax = thre
		self.plin_spl = plin
		self.k_max = 100
		# self.k_max = self.calc_kmax(thre)
		self.k_target = self.calc_ktarget()
		self.z_fid, self.plin_fid = self.find_fiducial(self.qs)
		if self.z_fid < -0.3:
			print (u'Warning: the amplitude of the target model is too high.')
			print (u'An appropriate fiducial model template for the reconstruction cannot be found.')
		self.pnl_fid = self.cfid.get_pnl(self.ks,self.z_fid)

	def compute_nsteps_default(self):
		return int(self.calc_normalized_distance()/0.08)

	def find_path(self,nste=-1,one_before=0):
		Om_best, self.plin_best, D2_best = self.get_closest_model()
		# print 'distance:',self.calc_normalized_distance()
		self.k_max = 100
		if nste < 0:
			self.nsteps = self.compute_nsteps_default()
		else:
			self.nsteps = nste
		print (u'number of intermediate steps:'), self.nsteps

		if self.nsteps > 0:
			s8 = get_sigmaR(8.,self.plin_spl)
			# print (u'sigma8=', s8)
			if s8 > 0.65:
				print (u'Warning: amplitude sigma8(z)=', s8, ' is high AND the target model is far.')
				print (u'The accuracy of the reconstruction is not guaranteed.' )

		Om_mid = np.linspace(0.3156,Om_best,2*self.nsteps+1)
		self.plin_diffs = []
		self.responses = []
		plin_now = self.plin_fid
		Om_now = 0.3156
		D2_now = self.D2_fid
		for n in range(1,2*self.nsteps+1):
			Om_now = Om_mid[n]
			D2_now, plin_mid = self.find_varied_Om(Om_now,self.qs)
			if n%2 == 0:
				self.plin_diffs.append(plin_mid-plin_now)
				plin_now = plin_mid
			else:
				self.responses.append(self.resp._get_response(D2_now,Om_now))
				plin_now_spl = InterpolatedUnivariateSpline(self.qs,plin_mid)
				kmax_tmp = self.calc_kmax(plin_now_spl)
				# print ('tmp:', Om_now, kmax_tmp)
				if self.k_max > kmax_tmp:
					self.k_max = kmax_tmp
				else:
					pass
		self.plin_diffs.append(self.plin_spl(self.qs)-plin_now)
		self.responses.append(self.resp._get_response(D2_now,Om_now))
		if one_before * self.nsteps:
                        pass
                else:
			plin_now_spl = InterpolatedUnivariateSpline(self.qs,plin_now)
			kmax_tmp = self.calc_kmax(plin_now_spl)
			# print ('tmp:', Om_now, kmax_tmp)
			if self.k_max > kmax_tmp:
				self.k_max = kmax_tmp
			else:
				pass



	def reconstruct(self):
		corr = np.zeros(201)
		for n in range(self.nsteps+1):
			for m in range(301):
				corr += self.responses[n][:,m] * self.plin_diffs[n][m]
		corr *= self.resp.deltak
		return self.pnl_fid + corr

	def calc_distance(self,p1,p2):
		w = self.qs<self.k_target
		return np.sum(((p1[w]-p2[w])/((p1[w]+p2[w])/2))**2)

	def calc_normalized_distance(self):
		w = self.qs<self.k_target
		# pt=self.plin_spl(self.qs[w])
		pt=self.plin_best[w]
		pf=self.plin_fid[w]
		return np.sqrt(np.sum(((pt-pf)/((pt+pf)/2.))**2)/w.size)

	def get_closest_model(self):
		Om_max = 0.5
		Om_min = 0.1
		dist_sqr_min = 1e10
		for Om_try in np.linspace(0.1,0.5,400):
			D2, plin_interm = self.find_varied_Om(Om_try,self.qs)
			dist_sqr = self.calc_distance(self.plin_spl(self.qs),plin_interm)
			if dist_sqr < dist_sqr_min:
				dist_sqr_min = dist_sqr
				Om_best = Om_try
				plin_best = plin_interm
				D2_best = D2
		return Om_best, plin_best, D2_best

	def find_fiducial(self,ks):
		nsample = 2**10 + 1
		k = np.logspace(-6,np.log10(self.k_target),nsample)
		integrand = k**self.tilt * self.cfid.get_plin(k,0.)
		dx = np.log(k[1])-np.log(k[0])
		sig = integrate.romb(integrand,dx)
		self.D2_fid = self.s_target/sig
		z = self.D2_to_z_fid(self.D2_fid)
		plin = self.D2_fid*self.cfid.get_plin(ks,0)
		return z, plin

	def D2_to_z_fid(self,D2):
		amin, amax = 0.0001, 10.
		while (amax-amin)/amin > 1e-5:
			atry = (amin+amax)/2.
			D2_try = linearGrowth(atry)**2
			if D2_try > D2:
				amax = atry
			else:
				amin = atry
		ascale = (amin+amax)/2.
		return 1./ascale - 1.

	def find_varied_Om(self,Om,ks):
		nsample = 2**10 + 1
		k = np.logspace(-6,np.log10(self.k_target),nsample)
		integrand = k**self.tilt * self.resp.get_plin_Om(k,0.,Om)
		dx = np.log(k[1])-np.log(k[0])
		sig = integrate.romb(integrand,dx)
		D2 = self.s_target/sig
		plin = D2*self.resp.get_plin_Om(ks,0.,Om)
		return D2, plin

	def calc_ktarget(self):
		kmin, kmax = 0.001, 10.
		nsample = 2**10 + 1
		while kmax-kmin>1e-5:
			kthre = (kmin+kmax)/2.
			k=np.logspace(-6,np.log10(kthre),nsample)
			dx = np.log(k[1])-np.log(k[0])
			integrand = k**self.tilt * self.plin_spl(k)
			sig = integrate.romb(integrand,dx)
			if sig > self.s_target:
				kmax = kthre
			else:
				kmin = kthre
		return (kmin+kmax)/2.

	def calc_kmax(self,plin_now_spl):
		kkmin, kkmax = 0.001, 10.
		while kkmax-kkmin>1e-4:
			kktry = (kkmax+kkmin)/2.
			alp_k =  self.get_sigmad_cut(plin_now_spl,kktry)* kktry**2 / 2.
			if alp_k > self.thre_kmax:
				kkmax = kktry
			else:
				kkmin = kktry
		return (kkmax+kkmin)/2.

	def get_sigmad_cut(self,plin_now_spl,k):
		return 1./(6.*np.pi**2) * integrate.quad(lambda t: plin_now_spl(t), 0.00001, k/2., limit=100, full_output=1)[0]

	def get_plin_fid(self,k):
		return self.cfid.get_plin(k,self.z_fid)

	def get_pnl_fid(self,k):
		return self.cfid.get_pnl(k,self.z_fid)

	def get_ktarget(self):
		return self.k_target

	def get_kmax(self):
		return self.k_max

	def get_kinternal(self):
		return self.ks

class ResponseData:
	def __init__(self):
		self.qs = np.linspace(0.0025,1.5025,301)
		self.ks = np.linspace(0.0025,1.0025,201)
		self.deltak = 0.005
		print ('Load precomputed data files...')
		self.prepare_st_data()
		self.prepare_kernel_data()

	def prepare_st_data(self):
		klist = np.load(os.path.dirname(__file__) + '/data/ks.npy')
		omlist = np.load(os.path.dirname(__file__) + '/data/oms.npy')

		p0_data = np.load(os.path.dirname(__file__) + '/data/st_p0.npy')
		g1_data = np.load(os.path.dirname(__file__) + '/data/st_g1.npy')
		g2_data = np.load(os.path.dirname(__file__) + '/data/st_g2.npy')
		p2corr_tt_data = np.load(os.path.dirname(__file__) + '/data/st_p2corr_tt.npy')
		p2corr_t1_data = np.load(os.path.dirname(__file__) + '/data/st_p2corr_t1.npy')
		p2corr_11_data = np.load(os.path.dirname(__file__) + '/data/st_p2corr_11.npy')
		p3corr_tt_data = np.load(os.path.dirname(__file__) + '/data/st_p3corr_tt.npy')
		alp_data = np.load(os.path.dirname(__file__) + '/data/st_alp.npy')

		self.p0_bispl = RectBivariateSpline(omlist,klist,p0_data)
		self.g1_bispl = RectBivariateSpline(omlist,klist,g1_data)
		self.g2_bispl = RectBivariateSpline(omlist,klist,g2_data)
		self.p2corr_tt_bispl = RectBivariateSpline(omlist,klist,p2corr_tt_data)
		self.p2corr_t1_bispl = RectBivariateSpline(omlist,klist,p2corr_t1_data)
		self.p2corr_11_bispl = RectBivariateSpline(omlist,klist,p2corr_11_data)
		self.p3corr_tt_bispl = RectBivariateSpline(omlist,klist,p3corr_tt_data)
		self.alp_bispl = RectBivariateSpline(omlist,klist,alp_data)

		# self.p0_bispl = joblib.load(os.path.dirname(__file__) + '/data/st_p0.pkl')
		# self.g1_bispl = joblib.load(os.path.dirname(__file__) + '/data/st_g1.pkl')
		# self.g2_bispl = joblib.load(os.path.dirname(__file__) + '/data/st_g2.pkl')
		# self.p2corr_tt_bispl = joblib.load(os.path.dirname(__file__) + '/data/st_p2corr_tt.pkl')
		# self.p2corr_t1_bispl = joblib.load(os.path.dirname(__file__) + '/data/st_p2corr_t1.pkl')
		# self.p2corr_11_bispl = joblib.load(os.path.dirname(__file__) + '/data/st_p2corr_11.pkl')
		# self.p3corr_tt_bispl = joblib.load(os.path.dirname(__file__) + '/data/st_p3corr_tt.pkl')
		# self.alp_bispl = joblib.load(os.path.dirname(__file__) + '/data/st_alp.pkl')

	def prepare_kernel_data(self):
		self.L1 = np.load(os.path.dirname(__file__) + '/data/kernel_L1.npy')
		self.M1 = np.load(os.path.dirname(__file__) + '/data/kernel_M1.npy')
		self.X2 = np.load(os.path.dirname(__file__) + '/data/kernel_X2.npy')
		self.Y2 = np.load(os.path.dirname(__file__) + '/data/kernel_Y2.npy')
		self.Q2 = np.load(os.path.dirname(__file__) + '/data/kernel_Q2.npy')
		self.S3 = np.load(os.path.dirname(__file__) + '/data/kernel_S3.npy')

	def Om_to_index(self,Om):
		return (Om - 0.1) / 0.01

	def get_SPT_kernels_z0(self,Om):
		Om_index = self.Om_to_index(Om)
		Om_array = Om_index * np.ones(201*301)
		k_dammy = np.arange(201)
		q_dammy = np.arange(301)
		kg, qg = np.meshgrid(k_dammy,q_dammy,indexing='ij')
		kg = np.ravel(kg)
		qg = np.ravel(qg)
		coords = np.array([Om_array,kg,qg])
		M1_SPT = ndimage.map_coordinates(self.M1,coords, order = 3).reshape(201,301)
		X2_SPT = ndimage.map_coordinates(self.X2,coords, order = 3).reshape(201,301)
		Y2_SPT = ndimage.map_coordinates(self.Y2,coords, order = 3).reshape(201,301)
		Q2_SPT = ndimage.map_coordinates(self.Q2,coords, order = 3).reshape(201,301)
		S3_SPT = ndimage.map_coordinates(self.S3,coords, order = 3).reshape(201,301)
		return (M1_SPT,X2_SPT,Y2_SPT,Q2_SPT,S3_SPT)

	def get_response(self,z,Om = 0.3156,wde = -1.):
		ascale = 1./(1.+z)
		D2 = linearGrowth(ascale,Om,wde)**2
		return self._get_response(D2,Om)

	def _get_response(self,D2,Om = 0.3156):
		D4 = D2**2
		D6 = D2*D4
		D8 = D2*D6
		p0 = D2*np.ravel(self.p0_bispl(Om,self.ks))
		g1 = D2*np.ravel(self.g1_bispl(Om,self.ks))
		g2 = D4*np.ravel(self.g2_bispl(Om,self.ks))
		alp_k = D2*np.ravel(self.alp_bispl(Om,self.ks))
		alp_q = D2*np.ravel(self.alp_bispl(Om,self.qs))

		p2corr_tt = D4*np.ravel(self.p2corr_tt_bispl(Om,self.ks))
		p2corr_t1 = D6*np.ravel(self.p2corr_t1_bispl(Om,self.ks))
		p2corr_11 = D8*np.ravel(self.p2corr_11_bispl(Om,self.ks))
		p3corr_tt = D6*np.ravel(self.p3corr_tt_bispl(Om,self.ks))
		M1_SPT, X2_SPT, Y2_SPT, Q2_SPT, S3_SPT  = self.get_SPT_kernels_z0(Om)
		M1_SPT *= D2
		X2_SPT *= D2
		Y2_SPT *= D4
		Q2_SPT *= D4
		S3_SPT *= D4
		kernel_full = np.zeros((201,301))
		# 1-loop
		for kbin in range(201):
			for qbin in range(301):
				kernel_full[kbin,qbin] = self.qs[qbin]**2/(2.*np.pi**2) * (2.*p0[kbin]*self.L1[kbin,qbin]+4.*X2_SPT[kbin,qbin])
		# 1-loop (diagonal) 
		for n in range(201):
			kernel_full[n,n] += 2. * g1[n] / self.deltak
		# 1-loop regularization
		for n in range(201):
			for m in range(301):
				kernel_full[n,m] *= 1. + alp_k[n] + alp_q[m]
		# 1-loop counter term
		for n in range(201):
			kernel_full[n,n] += (1. + 2. * alp_k[n] + 2.*alp_k[n]*alp_k[n]) / self.deltak
		# 2-loop
		for n in range(201):
			for m in range(301):
				kernel_full[n,m] += self.qs[m]*self.qs[m]/(2.*np.pi**2) * (2.*(g1[n]*self.L1[n,m]+2.*M1_SPT[n,m])*p0[n]+18.*S3_SPT[n,m]+8.*Y2_SPT[n,m]+4.*Q2_SPT[n,m])
		# 2-loop (diagonal)
		for n in range(201):
			kernel_full[n,n] += (g1[n]**2+2.*g2[n]) / self.deltak
		# 2-loop regularization
		for n in range(201):
			for m in range(301):
				if kernel_full[n,m] >0:
					kernel_full[n,m] *= np.exp(-alp_k[n]-alp_q[m])
				else:
					kernel_full[n,m] *= np.exp(-alp_k[n])/(1+alp_q[m])
		return kernel_full

	def get_plin_Om(self,k,z,Om):
		ascale = 1./(1.+z)
		D2 = linearGrowth(ascale,Om,-1.)**2
		return D2*np.ravel(self.p0_bispl(Om,k))

class FiducialCosmology:
	def __init__(self):
		zfid_data = np.load(os.path.dirname(__file__) + '/data/pnl_zfid.npy')
		kfid_data = np.load(os.path.dirname(__file__) + '/data/pnl_kfid.npy')
		pfid_data = np.load(os.path.dirname(__file__) + '/data/pnl_pfid.npy')
		self.pfid_spline = RectBivariateSpline(zfid_data[19::-1], kfid_data, np.log(pfid_data[19::-1,]))
		# self.pfid_spline = joblib.load(os.path.dirname(__file__) + '/data/cfid_pnl_spline.pkl')		
		self.cosmo = Cosmology(0.3156,0.6727,2.2065e-9,0.9645,0.05,-1.,os.path.dirname(__file__) + '/data/transfer/tkpl15.dat')

	def get_pnl(self,k,z):
		w = k < 0.01
		p1 = self.get_plin(k[w],z)
		w = k >= 0.01
		p2 = np.exp(self.pfid_spline(z,k[w]))
		return np.append(p1,p2)
	def get_plin(self,k,z):
		return self.cosmo.get_plin(k,z)

class Cosmology:
	def __init__(self,Om,h,As,ns,k0,w,tfname):
		self.cpara = CosmoParam(Om,h,As,ns,k0,w)
		tk = np.loadtxt(tfname).T
		tk[1,] *= (tk[0,]*h)**2
		tk_spline = InterpolatedUnivariateSpline(tk[0,],tk[1,])
		pzeta = (2.*np.pi**2)/tk[0,]**3 * As * (tk[0,]/(k0/h))**(ns-1.)
		self.plin_spline = InterpolatedUnivariateSpline(tk[0,],pzeta*tk[1,]**2)

	def get_plin(self,k,z):
		ascale = 1./(1.+z)
		D2 = linearGrowth(ascale,Om=self.cpara.Om,wde=self.cpara.w)**2
		return D2*self.plin_spline(k)

class CosmoParam:
	def __init__(self,Om,h,As,ns,k0,w):
		self.Om = Om
		self.h = h
		self.As = As
		self.ns = ns
		self.k0 = k0
		self.w = w

def linearGrowth(a,Om=0.3156,wde=-1):
	D = _linearGrowth(a,Om,wde)
	D0 = _linearGrowth(1,Om,wde)
	return D/D0;

def _linearGrowth(a,Om=0.3156,wde=-1):
	Ode = 1 - Om
	alpha = -1./(3.*wde)
	beta = (wde-1.)/(2.*wde)
	gamma = 1.-5./(6.*wde)
	x = -Ode/Om * a**(-3.*wde);
	res = integrate.quad(lambda t: t**(beta-1.)*(1.-t)**(gamma-beta-1.)*(1.-t*x)**(-alpha), 0, 1.)
	return a * res[0];

def THwindow(x):
	return 3.*(np.sin(x)-x*np.cos(x))/x**3

def get_sigmaR(R, plin_spline):
	nsample = 2**10 + 1
	k=np.logspace(-3,3,nsample)
	kR = k*R
	integrand = k**3 * plin_spline(k)/(2.*np.pi**2) * THwindow(kR)**2
	dx = np.log(k[1])-np.log(k[0])
	return integrate.romb(integrand,dx)**0.5
