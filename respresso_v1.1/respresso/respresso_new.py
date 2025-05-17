import os
import numpy as np
from scipy import ndimage, integrate
from scipy.interpolate import RectBivariateSpline, InterpolatedUnivariateSpline
import joblib

class respresso_core:
    def __init__(self):
        print('Hello. This is RESPRESSO.')
        self.resp = ResponseData()
        self.cfid = FiducialCosmology()
        self.qs = np.linspace(0.0025, 1.5025, 301)
        self.ks = np.linspace(0.0025, 1.0025, 201)
        print('RESPRESSO ready.')

    def set_target(self, plin, t=3, s=0.25, thre=1.):
        self.tilt = t
        self.s_target = s * 2. * np.pi**2
        self.thre_kmax = thre
        self.plin_spl = plin
        self.k_max = 100
        self.k_target = self.calc_ktarget()
        self.z_fid, self.plin_fid = self.find_fiducial(self.qs)
        if self.z_fid < -0.3:
            print('Warning: the amplitude of the target model is too high.')
            print('An appropriate fiducial model template for the reconstruction cannot be found.')
        self.pnl_fid = self.cfid.get_pnl(self.ks, self.z_fid)

    def compute_nsteps_default(self):
        return int(self.calc_normalized_distance() / 0.08)

    def find_path(self, nste=-1, one_before=False):
        Om_best, self.plin_best, D2_best = self.get_closest_model()
        self.k_max = 100
        self.nsteps = self.compute_nsteps_default() if nste < 0 else nste
        print('number of intermediate steps:', self.nsteps)

        if self.nsteps > 0:
            s8 = get_sigmaR(8., self.plin_spl)
            if s8 > 0.65:
                print('Warning: amplitude sigma8(z)=', s8, ' is high AND the target model is far.')
                print('The accuracy of the reconstruction is not guaranteed.')

        Om_mid = np.linspace(0.3156, Om_best, 2 * self.nsteps + 1)
        self.plin_diffs = []
        self.responses = []
        plin_now = self.plin_fid
        D2_now = self.D2_fid
        for n in range(1, 2 * self.nsteps + 1):
            Om_now = Om_mid[n]
            D2_now, plin_mid = self.find_varied_Om(Om_now, self.qs)
            if n % 2 == 0:
                self.plin_diffs.append(plin_mid - plin_now)
                plin_now = plin_mid
            else:
                self.responses.append(self.resp._get_response(D2_now, Om_now))
                plin_now_spl = InterpolatedUnivariateSpline(self.qs, plin_mid)
                kmax_tmp = self.calc_kmax(plin_now_spl)
                if self.k_max > kmax_tmp:
                    self.k_max = kmax_tmp
        self.plin_diffs.append(self.plin_spl(self.qs) - plin_now)
        self.responses.append(self.resp._get_response(D2_now, Om_mid[-1]))

        if not one_before:
            plin_now_spl = InterpolatedUnivariateSpline(self.qs, plin_now)
            kmax_tmp = self.calc_kmax(plin_now_spl)
            if self.k_max > kmax_tmp:
                self.k_max = kmax_tmp

    def reconstruct(self):
        corr = np.zeros_like(self.ks)
        for n in range(self.nsteps + 1):
            for m in range(self.qs.size):
                corr += self.responses[n][:, m] * self.plin_diffs[n][m]
        corr *= self.resp.deltak
        return self.pnl_fid + corr

    def calc_distance(self, p1, p2):
        mask = self.qs < self.k_target
        return np.sum(((p1[mask] - p2[mask]) / ((p1[mask] + p2[mask]) / 2))**2)

    def calc_normalized_distance(self):
        mask = self.qs < self.k_target
        pt = self.plin_best[mask]
        pf = self.plin_fid[mask]
        return np.sqrt(np.sum(((pt - pf) / ((pt + pf) / 2.))**2) / mask.sum())

    def get_closest_model(self):
        dist_min = np.inf
        for Om_try in np.linspace(0.1, 0.5, 400):
            D2, plin_mid = self.find_varied_Om(Om_try, self.qs)
            dist = self.calc_distance(self.plin_spl(self.qs), plin_mid)
            if dist < dist_min:
                dist_min = dist
                Om_best, plin_best, D2_best = Om_try, plin_mid, D2
        return Om_best, plin_best, D2_best

    def find_fiducial(self, ks):
        nsample = 2**10 + 1
        k = np.logspace(-6, np.log10(self.k_target), nsample)
        integrand = k**self.tilt * self.cfid.get_plin(k, 0.)
        dx = np.diff(np.log(k))[0]
        sig = integrate.romb(integrand, dx)
        self.D2_fid = self.s_target / sig
        z = self.D2_to_z_fid(self.D2_fid)
        plin = self.D2_fid * self.cfid.get_plin(ks, 0)
        return z, plin

    def D2_to_z_fid(self, D2):
        a_min, a_max = 0.0001, 10.
        while (a_max - a_min) / a_min > 1e-5:
            a_mid = 0.5 * (a_min + a_max)
            if _linearGrowth(a_mid)**2 > D2:
                a_max = a_mid
            else:
                a_min = a_mid
        a_scale = 0.5 * (a_min + a_max)
        return 1. / a_scale - 1.

    def find_varied_Om(self, Om, ks):
        nsample = 2**10 + 1
        k = np.logspace(-6, np.log10(self.k_target), nsample)
        integrand = k**self.tilt * self.resp.get_plin_Om(k, 0., Om)
        dx = np.diff(np.log(k))[0]
        sig = integrate.romb(integrand, dx)
        D2 = self.s_target / sig
        plin = D2 * self.resp.get_plin_Om(ks, 0., Om)
        return D2, plin

    def calc_ktarget(self):
        k_min, k_max = 0.001, 10.
        nsample = 2**10 + 1
        while k_max - k_min > 1e-5:
            k_mid = 0.5 * (k_min + k_max)
            k_arr = np.logspace(-6, np.log10(k_mid), nsample)
            dx = np.diff(np.log(k_arr))[0]
            sig = integrate.romb(k_arr**self.tilt * self.plin_spl(k_arr), dx)
            if sig > self.s_target:
                k_max = k_mid
            else:
                k_min = k_mid
        return 0.5 * (k_min + k_max)

    def calc_kmax(self, plin_spl):
        kmn, kmx = 0.001, 10.
        while kmx - kmn > 1e-4:
            k_mid = 0.5 * (kmx + kmn)
            alp = self.get_sigmad_cut(plin_spl, k_mid) * k_mid**2 / 2.
            if alp > self.thre_kmax:
                kmx = k_mid
            else:
                kmn = k_mid
        return 0.5 * (kmx + kmn)

    def get_sigmad_cut(self, plin_spl, k):
        val, _ = integrate.quad(plin_spl, 0.00001, k/2., limit=100)
        return val / (6. * np.pi**2)

    def get_plin_fid(self, k):
        return self.cfid.get_plin(k, self.z_fid)

    def get_pnl_fid(self, k):
        return self.cfid.get_pnl(k, self.z_fid)

    def get_ktarget(self):
        return self.k_target

    def get_kmax(self):
        return self.k_max

    def get_kinternal(self):
        return self.ks

class ResponseData:
    def __init__(self):
        print('Load precomputed data files...')
        self.qs = np.linspace(0.0025, 1.5025, 301)
        self.ks = np.linspace(0.0025, 1.0025, 201)
        self.deltak = 0.005
        self.prepare_st_data()
        self.prepare_kernel_data()

    def prepare_st_data(self):
        base = os.path.dirname(__file__) + '/data'
        klist = np.load(f'{base}/ks.npy')
        omlist = np.load(f'{base}/oms.npy')

        data_files = ['st_p0', 'st_g1', 'st_g2', 'st_p2corr_tt', 'st_p2corr_t1',
                      'st_p2corr_11', 'st_p3corr_tt', 'st_alp']
        bisplines = {}
        for fname in data_files:
            arr = np.load(f'{base}/{fname}.npy')
            bisplines[fname] = RectBivariateSpline(omlist, klist, arr)

        self.p0_bispl = bisplines['st_p0']
        self.g1_bispl = bisplines['st_g1']
        self.g2_bispl = bisplines['st_g2']
        self.p2corr_tt_bispl = bisplines['st_p2corr_tt']
        self.p2corr_t1_bispl = bisplines['st_p2corr_t1']
        self.p2corr_11_bispl = bisplines['st_p2corr_11']
        self.p3corr_tt_bispl = bisplines['st_p3corr_tt']
        self.alp_bispl = bisplines['st_alp']

    def prepare_kernel_data(self):
        base = os.path.dirname(__file__) + '/data'
        self.L1 = np.load(f'{base}/kernel_L1.npy')
        self.M1 = np.load(f'{base}/kernel_M1.npy')
        self.X2 = np.load(f'{base}/kernel_X2.npy')
        self.Y2 = np.load(f'{base}/kernel_Y2.npy')
        self.Q2 = np.load(f'{base}/kernel_Q2.npy')
        self.S3 = np.load(f'{base}/kernel_S3.npy')

    def Om_to_index(self, Om):
        return (Om - 0.1) / 0.01

    def get_SPT_kernels_z0(self, Om):
        idx = self.Om_to_index(Om)
        grid_k, grid_q = np.meshgrid(np.arange(201), np.arange(301), indexing='ij')
        coords = np.vstack([np.full(grid_k.size, idx), grid_k.ravel(), grid_q.ravel()])
        M1 = ndimage.map_coordinates(self.M1, coords, order=3).reshape(201, 301)
        X2 = ndimage.map_coordinates(self.X2, coords, order=3).reshape(201, 301)
        Y2 = ndimage.map_coordinates(self.Y2, coords, order=3).reshape(201, 301)
        Q2 = ndimage.map_coordinates(self.Q2, coords, order=3).reshape(201, 301)
        S3 = ndimage.map_coordinates(self.S3, coords, order=3).reshape(201, 301)
        return M1, X2, Y2, Q2, S3

    def get_response(self, z, Om=0.3156, wde=-1.):
        D2 = linearGrowth(1./(1.+z), Om, wde)**2
        return self._response(D2, Om)
    
    def _get_response(self, D2, Om=0.3156):
        D4, D6, D8 = D2**2, D2**3, D2**4
        p0 = D2 * self.p0_bispl(Om, self.ks).ravel()
        g1 = D2 * self.g1_bispl(Om, self.ks).ravel()
        g2 = D4 * self.g2_bispl(Om, self.ks).ravel()
        alp_k = D2 * self.alp_bispl(Om, self.ks).ravel()
        alp_q = D2 * self.alp_bispl(Om, self.qs).ravel()

        tt = D4 * self.p2corr_tt_bispl(Om, self.ks).ravel()
        t1 = D6 * self.p2corr_t1_bispl(Om, self.ks).ravel()
        _11 = D8 * self.p2corr_11_bispl(Om, self.ks).ravel()
        p3 = D6 * self.p3corr_tt_bispl(Om, self.ks).ravel()

        M1, X2, Y2, Q2, S3 = self.get_SPT_kernels_z0(Om)
        M1 *= D2; X2 *= D2; Y2 *= D4; Q2 *= D4; S3 *= D4

        kernel = np.zeros((201, 301))
        for i in range(201):
            for j in range(301):
                kernel[i, j] = self.qs[j]**2/(2.*np.pi**2)*(2.*p0[i]*self.L1[i, j] + 4.*X2[i, j])
        for i in range(201):
            kernel[i, i] += 2.*g1[i]/self.deltak
        for i in range(201):
            for j in range(301):
                kernel[i, j] *= 1. + alp_k[i] + alp_q[j]
        for i in range(201):
            kernel[i, i] += (1. + 2.*alp_k[i] + 2.*alp_k[i]**2)/self.deltak
        for i in range(201):
            for j in range(301):
                kernel[i, j] += self.qs[j]**2/(2.*np.pi**2)*(2.*(g1[i]*self.L1[i,j] + 2.*M1[i,j])*p0[i] + 18.*S3[i,j] + 8.*Y2[i,j] + 4.*Q2[i,j])
        for i in range(201):
            kernel[i, i] += (g1[i]**2 + 2.*g2[i])/self.deltak
        for i in range(201):
            for j in range(301):
                kernel[i, j] *= np.exp(-alp_k[i] - alp_q[j]) if kernel[i, j] > 0 else np.exp(-alp_k[i])/(1 + alp_q[j])
        return kernel

    def get_plin_Om(self, k, z, Om):
        D2 = linearGrowth(1./(1.+z), Om, -1.)**2
        return D2 * self.p0_bispl(Om, k).ravel()

class FiducialCosmology:
    def __init__(self):
        base = os.path.dirname(__file__) + '/data'
        zfid = np.load(f'{base}/pnl_zfid.npy')
        kfid = np.load(f'{base}/pnl_kfid.npy')
        pfid = np.load(f'{base}/pnl_pfid.npy')
        self.pfid_spline = RectBivariateSpline(zfid[::-1], kfid, np.log(pfid[::-1, :]))
        self.cosmo = Cosmology(0.3156, 0.6727, 2.2065e-9, 0.9645, 0.05, -1., f'{base}/transfer/tkpl15.dat')

    def get_pnl(self, k, z):
        mask = k < 0.01
        p1 = self.get_plin(k[mask], z)
        p2 = np.exp(self.pfid_spline(z, k[~mask])).ravel()
        return np.concatenate((p1, p2))

    def get_plin(self, k, z):
        return self.cosmo.get_plin(k, z)

class Cosmology:
    def __init__(self, Om, h, As, ns, k0, w, tfname):
        self.cpara = CosmoParam(Om, h, As, ns, k0, w)
        tk = np.loadtxt(tfname).T
        tk[1] *= (tk[0]*h)**2
        spline = InterpolatedUnivariateSpline(tk[0], tk[1])
        pzeta = (2.*np.pi**2)/tk[0]**3 * As * (tk[0]/(k0/h))**(ns-1)
        self.plin_spline = InterpolatedUnivariateSpline(tk[0], pzeta * tk[1]**2)

    def get_plin(self, k, z):
        D2 = linearGrowth(1./(1.+z), self.cpara.Om, self.cpara.w)**2
        return D2 * self.plin_spline(k)

class CosmoParam:
    def __init__(self, Om, h, As, ns, k0, w):
        self.Om, self.h, self.As, self.ns, self.k0, self.w = Om, h, As, ns, k0, w

def linearGrowth(a, Om=0.3156, wde=-1):
    D = _linearGrowth(a, Om, wde)
    D0 = _linearGrowth(1, Om, wde)
    return D / D0

def _linearGrowth(a, Om=0.3156, wde=-1):
    Ode = 1 - Om
    alpha = -1./(3.*wde)
    beta = (wde-1.)/(2.*wde)
    gamma = 1.-5./(6.*wde)
    x = -Ode/Om * a**(-3.*wde)
    res, _ = integrate.quad(lambda t: t**(beta-1)*(1.-t)**(gamma-beta-1)*(1.-t*x)**(-alpha), 0, 1)
    return a * res

def THwindow(x):
    return 3.*(np.sin(x)-x*np.cos(x))/x**3

def get_sigmaR(R, plin_spl):
    nsample = 2**10 + 1
    k = np.logspace(-3, 3, nsample)
    integrand = k**3 * plin_spl(k)/(2.*np.pi**2) * THwindow(k*R)**2
    dx = np.diff(np.log(k))[0]
    return integrate.romb(integrand, dx)**0.5
