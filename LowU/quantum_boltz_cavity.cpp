// quantum_boltz_cavity.cpp
// Steady-state Keldysh DMFT for the Hubbard model coupled to a driven-dissipative cavity
// 2-site unit cell, 1D chain (extendable to 2D via transverse dispersion)
//
// Self-consistent perturbation theory:
//   - 2nd order (IPT) Hubbard self-energy, element-wise in sublattice indices
//   - Cavity-mediated Fock self-energy (non-equilibrium, no FDT)
//   - Cavity Hartree (self-consistent phi)
//   - Explicit k-summation for G_loc (no Bethe lattice trick)
//   - Markovian bath for broadening

#include <iostream>
#include <iomanip>
#include <fstream>
#include <complex>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <sstream>
#include "omp.h"
#include "./find_param.h"
#include "./ness_decls.hpp"

using namespace std;
using namespace ness;

// ================================================================
// Utility functions
// ================================================================

void ForceImag(GF &G) {
    for (long i = 0; i < G.ngrid_; i++) {
        cdmatrix tmp = G.Lesser[i];
        G.Lesser[i] = 0.5 * (tmp - tmp.adjoint());
    }
}

double fermi(double w, double beta) {
    double arg = w * beta;
    if (arg * arg > 10000.0) return (w > 0 ? 0.0 : 1.0);
    return 1.0 / (exp(arg) + 1.0);
}

void force_equi(GF &G, double beta) {
    for (long w = 0; w < G.ngrid_; w++) {
        double omega = G.grid_[w];
        double fw = fermi(omega, beta);
        G.Lesser[w] = fw * (G.Retarded[w].adjoint() - G.Retarded[w]);
    }
}

void mixing_G(GF &Gnew, GF &Gold, double mix) {
    for (int w = 0; w < Gnew.ngrid_; w++) {
        Gnew.Retarded[w] = Gnew.Retarded[w] * (1.0 - mix) + Gold.Retarded[w] * mix;
        Gnew.Lesser[w] = Gnew.Lesser[w] * (1.0 - mix) + Gold.Lesser[w] * mix;
    }
}

template <class Matrix>
void DensityMatrix(const GF &G, Matrix &M) {
    cplx ii(0.0, 1.0);
    M.resize(G.size1_, G.size2_);
    M.setZero();
    for (long i = 0; i < G.ngrid_; i++) M += G.Lesser[i];
    M = M * G.dgrid_ / (2.0 * M_PI * ii);
}

// Markovian bath: constant broadening, equilibrium FDT
void setBath_diag(double Jcoupl, double beta, GF &Bath) {
    cplx imag_one(0, 1);
    cdmatrix Ret(Bath.size1_, Bath.size1_);
    Ret.setZero();
    for (int j = 0; j < Bath.size1_; j++) Ret(j, j) = -imag_one * Jcoupl;
    for (long w = 0; w < Bath.ngrid_; w++) Bath.Retarded[w] = Ret;
    force_equi(Bath, beta);
}

// ================================================================
// tau_z * M * tau_z for a 2x2 matrix (flips sign of off-diagonal)
// ================================================================

inline cdmatrix tau_z_sandwich(const cdmatrix &M) {
    cdmatrix R(2, 2);
    R(0, 0) = M(0, 0);
    R(0, 1) = -M(0, 1);
    R(1, 0) = -M(1, 0);
    R(1, 1) = M(1, 1);
    return R;
}

// ================================================================
// Cavity propagator (symmetrized, scalar)
//
// D~^R(w) = delta / ((w + i*Gamma/2)^2 - delta^2)
// D~^K(w) = -i*Gamma/2 * [1/((w-delta)^2 + Gamma^2/4)
//                        + 1/((w+delta)^2 + Gamma^2/4)]
// D^< = (D^K - D^R + D^A) / 2,  with D^A = conj(D^R)
//
// eval_D evaluates D^R, D^K at an arbitrary (generally off-grid) frequency,
// so that the analytic cavity self-energy below never needs to interpolate
// the tabulated Dcav grid.
// ================================================================

void eval_D(double omega, double delta, double Gamma, cplx &DR, cplx &DK) {
    cplx ii(0.0, 1.0);
    double G2 = Gamma / 2.0;
    double G2sq = G2 * G2;

    cplx z = omega + ii * G2;
    DR = delta / (z * z - delta * delta);

    double d1 = (omega - delta) * (omega - delta) + G2sq;
    double d2 = (omega + delta) * (omega + delta) + G2sq;
    DK = -ii * G2 * (1.0 / d1 + 1.0 / d2);
}

// Same D^R (causal structure fixed by delta, Gamma) but with D^K replaced by
// its EXACT thermal/FDT value at inverse temperature beta:
//   D^K(w) = coth(beta*w/2) * (D^R(w) - D^A(w))
// Used only for the thermal-FDT regression test below.
void eval_D_thermal(double omega, double delta, double Gamma, double beta,
                    cplx &DR, cplx &DK) {
    cplx DK_unused;
    eval_D(omega, delta, Gamma, DR, DK_unused);
    cplx DA = conj(DR);
    double freq = (abs(omega) < 1e-12) ? 1e-12 : omega;   // removes division by zero issues
    double coth_x = 1.0 / tanh(beta * freq / 2.0);
    DK = coth_x * (DR - DA);  // Equilibrium FDT
}

void set_cavity_propagator(double delta, double Gamma, GF &Dcav) {
    for (long w = 0; w < Dcav.ngrid_; w++) {
        double omega = Dcav.grid_[w];

        cplx DR, DK;
        eval_D(omega, delta, Gamma, DR, DK);
        cplx DA = conj(DR);
        cplx DL = (DK - DR + DA) / 2.0;

        Dcav.Retarded[w](0, 0) = DR;
        Dcav.Lesser[w](0, 0) = DL;
    }
}

// ================================================================
// Analytic cavity self-energy (sharp-quasiparticle / full k-dependence)
//
// For each k, h_k + Hartree is diagonalized into two quasiparticle bands
// nu = -,+ with energy eps_{k,nu} and projector P_{k,nu}. The cavity
// self-energy is diagonal in k (q=0 boson exchange) and, assuming sharp
// quasiparticles, has the closed form (F = distribution function,
// D~ = symmetrized cavity propagator from eval_D):
//
//   Sigma^R_cav(k,w) - Sigma^A_cav(k,w) =
//     g_eff_sq * sum_nu { D~^K(w-eps_{k,nu})
//                        + F(eps_{k,nu}) * [D~^R(w-eps_{k,nu}) - D~^A(w-eps_{k,nu})] }
//                * tz P_{k,nu} tz
//
//   Sigma^<_cav(k,w) =
//     -g_eff_sq * sum_nu { D~^K(w-eps_{k,nu})
//                        - [D~^R(w-eps_{k,nu}) - D~^A(w-eps_{k,nu})] }
//                * (1 - F(eps_{k,nu}))/2 * tz P_{k,nu} tz
//
// (the overall minus sign on Sigma^<_cav, absent from the literal formula, is
// an empirically-confirmed correction: without it, feeding the exact
// equilibrium F(eps)=tanh(beta*eps/2) into the Dyson equation with no other
// source of damping gives a sign-inverted, unphysical density; with it,
// density converges to the correct value -- see conversation history for the
// isolation of this sign, most likely a Keldysh-contour sign convention
// mismatch between the original derivation and this code's G^</G^R
// conventions).
//
// Sigma^R_cav is then taken as half of the antihermitian combination above
// (i.e. only the antihermitian/dissipative part is kept, matching the
// convention used previously for the FFT-based cavity self-energy).
//
// No FFT and no full k-resolved GF storage are needed: only the O(nk)
// band energies/projectors and the O(nk) values of F at those energies.
// ================================================================

struct BandData {
    array<double, 2> eps;   // eps[0] = eps_{k,-}, eps[1] = eps_{k,+}
    array<cdmatrix, 2> P;   // projectors onto the two bands
};

// Diagonalize h_k + Hartree at every k (h_k + Hartree is Hermitian, and
// since eps_perp = mu = 0 it is traceless so eigenvalues come out as -E,+E).
void compute_bands_k(const vector<cdmatrix> &hk, const cdmatrix &Hartree,
                     vector<BandData> &bands) {
    int nk = hk.size();
    bands.resize(nk);
    for (int k = 0; k < nk; k++) {
        cdmatrix Hk = hk[k] + Hartree;
        Eigen::SelfAdjointEigenSolver<cdmatrix> es(Hk);
        for (int nu = 0; nu < 2; nu++) {
            bands[k].eps[nu] = es.eigenvalues()(nu);
            cdmatrix v = es.eigenvectors().col(nu);
            bands[k].P[nu] = v * v.adjoint();
        }
    }
}

// tz P_{k,nu} tz for every k, nu (independent of omega, computed once per iteration)
vector<array<cdmatrix, 2>> compute_TPT(const vector<BandData> &bands) {
    vector<array<cdmatrix, 2>> TPT(bands.size());
    for (size_t k = 0; k < bands.size(); k++)
        for (int nu = 0; nu < 2; nu++)
            TPT[k][nu] = tau_z_sandwich(bands[k].P[nu]);
    return TPT;
}

// Extract a single, k-independent distribution function F(eps_{k,nu}) from
// the local GF via
//   (1-F(w))/2 = - Im tr Gloc^<(w) / (2 Im tr Gloc^R(w))
// (the minus sign matches this code's G^< = f*(G^A-G^R) convention, f the
// physical occupation, so that (1-F)/2 = f as needed by the Sigma_cav
// formulas above), binned around each of the 2*nk quasiparticle energies
// eps_{k,nu} (the omega-grid is finer than the k-grid, so several omega
// points contribute to each bin).
//
// The dispersion has van Hove points (dEps/dk -> 0) at k=0 and k=pi, so many
// k (not just the exact k<->-k pair) pile up into an energy window narrower
// than the omega-grid spacing near the band edges -- not just exact
// degeneracies. So rather than merging by an energy tolerance, neighbouring
// raw bins (one per sorted eps_{k,nu}, edges at midpoints between sorted
// neighbours) are greedily merged, left to right, until each accumulated
// group has at least min_pts omega-grid points -- this handles exact
// degeneracies (from k<->-k) and van Hove crowding uniformly, since both
// just mean "too few omega points in this energy window". Omega points
// beyond the two extremal (symmetric-width) edges are dropped as
// incoherent/off-shell weight.
void extract_F_binned(const GF &Gloc, const vector<BandData> &bands,
                      vector<array<double, 2>> &F_k) {
    int nk = bands.size();
    int M = 2 * nk;
    F_k.assign(nk, {0.5, 0.5});
    if (M == 0) return;

    vector<double> E(M);
    vector<pair<int, int>> tag(M);
    int idx = 0;
    for (int k = 0; k < nk; k++)
        for (int nu = 0; nu < 2; nu++) {
            E[idx] = bands[k].eps[nu];
            tag[idx] = make_pair(k, nu);
            idx++;
        }

    vector<int> order(M);
    iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [&](int a, int b) { return E[a] < E[b]; });

    vector<double> Es(M);
    for (int i = 0; i < M; i++) Es[i] = E[order[i]];

    vector<double> edges(M + 1);
    for (int i = 1; i < M; i++) edges[i] = 0.5 * (Es[i - 1] + Es[i]);
    if (M > 1) {
        edges[0] = Es[0] - 0.5 * (Es[1] - Es[0]);
        edges[M] = Es[M - 1] + 0.5 * (Es[M - 1] - Es[M - 2]);
    } else {
        edges[0] = Es[0] - 1.0;
        edges[M] = Es[0] + 1.0;
    }

    long N = Gloc.ngrid_;
    vector<double> omega(N), num(N), den(N);
    for (long w = 0; w < N; w++) {
        omega[w] = Gloc.grid_[w];
        num[w] = Gloc.Lesser[w].trace().imag();
        den[w] = Gloc.Retarded[w].trace().imag();
    }
    vector<long> wo(N);
    iota(wo.begin(), wo.end(), 0);
    sort(wo.begin(), wo.end(), [&](long a, long b) { return omega[a] < omega[b]; });
    vector<double> sortedOmega(N);
    for (long ii = 0; ii < N; ii++) sortedOmega[ii] = omega[wo[ii]];

    vector<double> numSum(M, 0.0), denSum(M, 0.0);
    vector<int> count(M, 0);

    int bin = 0;
    for (long ii = 0; ii < N; ii++) {
        double om = sortedOmega[ii];
        if (om < edges[0]) continue;
        if (om >= edges[M]) break;
        while (bin < M - 1 && om >= edges[bin + 1]) bin++;
        long w = wo[ii];
        numSum[bin] += num[w];
        denSum[bin] += den[w];
        count[bin]++;
    }

    const int min_pts = 5;
    vector<double> Fsorted(M);
    int n_out_of_range = 0;
    double worst = 0.0;
    int n_groups = 0;

    int i = 0;
    while (i < M) {
        int j = i;
        long accCount = count[i];
        double accNum = numSum[i], accDen = denSum[i];
        while (accCount < min_pts && j < M - 1) {
            j++;
            accCount += count[j];
            accNum += numSum[j];
            accDen += denSum[j];
        }

        double ratio;
        if (accCount > 0 && abs(accDen) > 1e-14) {
            ratio = accNum / accDen;
        } else {
            // Still nothing (can happen only right at the very edge of the
            // omega grid): fall back to the single nearest grid point.
            double center = 0.5 * (Es[i] + Es[j]);
            auto it = lower_bound(sortedOmega.begin(), sortedOmega.end(), center);
            long best = wo[min((long)(it - sortedOmega.begin()), N - 1)];
            if (it != sortedOmega.begin()) {
                long prevIdx = wo[(it - sortedOmega.begin()) - 1];
                if (abs(omega[prevIdx] - center) < abs(omega[best] - center)) best = prevIdx;
            }
            ratio = (abs(den[best]) > 1e-14) ? num[best] / den[best] : 0.0;
        }
        // Note: the code's convention (force_equi/DensityMatrix) gives
        // G^< = f*(G^A-G^R) with f the physical occupation, so identically
        // num/den = Im tr G^< / Im tr G^R = -2f (no extra factor of 2 beyond
        // this -- ratio here is num/den, not num/(2 den)). F is defined such
        // that (1-F)/2 = f, i.e. F = 1-2f = 1 + ratio.
        double Fval = 1.0 + ratio;

        n_groups++;
        if (Fval < -1.0 || Fval > 1.0) {
            n_out_of_range++;
            double excess = max(Fval - 1.0, -1.0 - Fval);
            if (excess > worst) worst = excess;
            // F is a physical occupation-like quantity, bounded in [-1,1] by
            // construction (and the Sigma_cav formulas are only guaranteed
            // causal for F in that range, see eval_Sigma_cav_k). An estimate
            // outside it is a resolution artifact (too few omega points, or
            // this bin straddling incoherent weight), not real physics --
            // clamp rather than feed an acausal Sigma_cav back into Gloc.
            Fval = max(-1.0, min(1.0, Fval));
        }
        for (int m = i; m <= j; m++) Fsorted[m] = Fval;

        i = j + 1;
    }

    for (int m = 0; m < M; m++) {
        int k = tag[order[m]].first;
        int nu = tag[order[m]].second;
        F_k[k][nu] = Fsorted[m];
    }

    if (n_out_of_range > 0) {
        cout << "  [extract_F_binned] WARNING: " << n_out_of_range << "/" << n_groups
             << " F(eps_k,nu) groups outside [-1,1] (max excess = " << worst << ")" << endl;
    }
}

// Direct (non-binned) extraction of F(eps_{k,nu}): linearly interpolate
// Im tr G^<_loc(w) and Im tr G^R_loc(w) separately onto each (generally
// off-grid) eps_{k,nu), then F = 1 + Im(interp G^<)/Im(interp G^R) (same
// sign/normalization as extract_F_binned above). No bin construction at all
// -- for nk large enough that the pointwise ratio curve is smooth, this is
// simpler and avoids the degeneracy/van-Hove-crowding/edge-bin machinery
// that binning needs. eps_{k,nu} outside the omega grid are clamped to the
// nearest edge point.
void extract_F_direct(const GF &Gloc, const vector<BandData> &bands,
                      vector<array<double, 2>> &F_k) {
    int nk = bands.size();
    F_k.assign(nk, {0.5, 0.5});
    if (nk == 0) return;

    long N = Gloc.ngrid_;
    vector<double> omega(N), num(N), den(N);
    for (long w = 0; w < N; w++) {
        omega[w] = Gloc.grid_[w];
        num[w] = Gloc.Lesser[w].trace().imag();
        den[w] = Gloc.Retarded[w].trace().imag();
    }
    vector<long> wo(N);
    iota(wo.begin(), wo.end(), 0);
    sort(wo.begin(), wo.end(), [&](long a, long b) { return omega[a] < omega[b]; });
    vector<double> sortedOmega(N);
    for (long ii = 0; ii < N; ii++) sortedOmega[ii] = omega[wo[ii]];

    int n_out_of_range = 0;
    double worst = 0.0;

    for (int k = 0; k < nk; k++) {
        for (int nu = 0; nu < 2; nu++) {
            double e = bands[k].eps[nu];
            double numI, denI;

            if (e <= sortedOmega.front()) {
                numI = num[wo.front()];
                denI = den[wo.front()];
            } else if (e >= sortedOmega.back()) {
                numI = num[wo.back()];
                denI = den[wo.back()];
            } else {
                auto it = lower_bound(sortedOmega.begin(), sortedOmega.end(), e);
                long i1 = it - sortedOmega.begin();
                long i0 = i1 - 1;
                double w0 = sortedOmega[i0], w1 = sortedOmega[i1];
                double t = (w1 > w0) ? (e - w0) / (w1 - w0) : 0.0;
                long a0 = wo[i0], a1 = wo[i1];
                numI = num[a0] + t * (num[a1] - num[a0]);
                denI = den[a0] + t * (den[a1] - den[a0]);
            }

            double ratio = (abs(denI) > 1e-14) ? numI / denI : 0.0;
            double Fval = 1.0 + ratio;

            if (Fval < -1.0 || Fval > 1.0) {
                n_out_of_range++;
                double excess = max(Fval - 1.0, -1.0 - Fval);
                if (excess > worst) worst = excess;
                Fval = max(-1.0, min(1.0, Fval)); // see extract_F_binned: physical bound
            }
            F_k[k][nu] = Fval;
        }
    }

    if (n_out_of_range > 0) {
        cout << "  [extract_F_direct] WARNING: " << n_out_of_range << "/" << (2 * nk)
             << " F(eps_k,nu) values outside [-1,1] (max excess = " << worst << ")" << endl;
    }
}

// Sigma^R_cav(k,w), Sigma^<_cav(k,w) from the analytic formulas above,
// for a single k (TPT[nu] = tz P_{k,nu} tz, F[nu] = F(eps_{k,nu})).
// beta_thermal < 0 (default): use the real driven-dissipative D (eval_D).
// beta_thermal >= 0: FDT-regression-test mode -- use eval_D_thermal instead,
// i.e. force the photon to be exactly thermal at that inverse temperature.
void eval_Sigma_cav_k(double omega, const array<double, 2> &eps,
                      const array<cdmatrix, 2> &TPT, const array<double, 2> &F,
                      double g_eff_sq, double delta_cav, double Gamma_cav,
                      cdmatrix &SigCavR, cdmatrix &SigCavL,
                      double beta_thermal = -1.0) {
    SigCavR = cdmatrix::Zero(2, 2);
    SigCavL = cdmatrix::Zero(2, 2);
    if (g_eff_sq < 1e-15) return; // in place modification of SigCavL/R

    cdmatrix sumRA = cdmatrix::Zero(2, 2);
    cdmatrix sumL = cdmatrix::Zero(2, 2);
    for (int nu = 0; nu < 2; nu++) {
        double freq = omega - eps[nu];
        cplx DR, DK;
        if (beta_thermal >= 0.0) eval_D_thermal(freq, delta_cav, Gamma_cav, beta_thermal, DR, DK);
        else eval_D(freq, delta_cav, Gamma_cav, DR, DK);
        cplx DA = conj(DR);

        cplx coeffRA = DK + F[nu] * (DR - DA);
        cplx coeffL = (DK - (DR - DA)) * (0.5 * ( F[nu] - 1.0 ));

        sumRA += coeffRA * TPT[nu];
        sumL += coeffL * TPT[nu];
    }

    SigCavR = 0.5 * g_eff_sq * sumRA;
    {
        // NOTE: must copy into a temporary before using .adjoint() here --
        // "X = 0.5*(X - X.adjoint())" is an Eigen self-aliasing hazard
        // (.adjoint() = .transpose().conjugate(), same index-permutation-
        // under-self-assignment issue as the GkR = 0.5*(GkR+GkR.transpose())
        // bug found in compute_gloc_ksum).
        cdmatrix tmp = SigCavR;
        SigCavR = 0.5 * (tmp - tmp.adjoint()); // enforce antihermiticity numerically
    }
    SigCavL =   g_eff_sq * sumL; //  0.5 *  might be required, I have to double check the calculations.
    {
        cdmatrix tmp = SigCavL;
        SigCavL = 0.5 * (tmp - tmp.adjoint());
    }
}

// ================================================================
// Hubbard self-energy: 2nd order, element-wise in sublattice indices
//
// [Sigma^<]_{ab}(t)  = (G^<_{ab})^2 * (-conj(G^>_{ab}))
// [Sigma^R]_{ab}(t)  = (G^>_{ab})^2 * (-conj(G^<_{ab})) - Sigma^<_{ab}
//
// This uses G^>_{ab}(-t) = -conj(G^>_{ab}(t)) (steady-state relation)
// Multiply by U^2 at the end.
// ================================================================

void compute_Sigma_hubbard(double U, const GF &Gloc, GF &Sigma_hub,
                           fft_solver &solver) {
    if (abs(U) < 1e-15) {
        Sigma_hub.clear();
        return;
    }

    long N = Gloc.ngrid_;
    long Nt = N * 3 / 2 + 1;

    GF tG(Gloc.dgrid_, Nt, 2, time_gf);
    GF tSigma(Gloc.dgrid_, Nt, 2, time_gf);

    solver.to_time(tG, Gloc, 1);

    for (long t = 0; t < Nt; t++) {
        for (int a = 0; a < 2; a++) {
            for (int b = 0; b < 2; b++) {
                cplx Gl = tG.Lesser[t](a, b);
                cplx Gg = tG.Greater(t)(a, b);

                cplx SL = Gl * Gl * (-conj(Gg));
                cplx SR = Gg * Gg * (-conj(Gl)) - SL;

                tSigma.Lesser[t](a, b) = SL;
                tSigma.Retarded[t](a, b) = SR;
            }
        }
    }

    tSigma.smul(U * U);
    tSigma.reset_grid(tG.dgrid_);
    // Greater(0) returns true G^>(0+), so products at t=0 are true values;
    // apply factor 0.5 to match the FFT convention for retarded at t=0
    tSigma.Retarded[0] *= 0.5;
    tSigma.Retarded[Nt - 1].setZero();

    solver.to_freq(Sigma_hub, tSigma);
    ForceImag(Sigma_hub);
}

// ================================================================
// Cavity Hartree (Hubbard Hartree set to 0)
//
// phi = -2*g*delta / (delta^2 + Gamma^2/4) * (n_0 - n_1)
// Hartree = g * phi * tau_z
//
// where n_a = Re(rho_{aa}) from the density matrix.
// ================================================================

void compute_hartree(double g_cav, double delta, double Gamma,
                     const GF &Gloc, cdmatrix &Hartree) {
    Hartree.setZero();
    if (abs(g_cav) < 1e-15) return;

    cdmatrix rho(2, 2);
    DensityMatrix(Gloc, rho);
    double n0 = rho(0, 0).real();
    double n1 = rho(1, 1).real();
    double delta_n = n0 - n1;

    double phi = -2.0 * g_cav * delta / (delta * delta + Gamma * Gamma / 4.0) * delta_n;

    Hartree(0, 0) = g_cav * phi;
    Hartree(1, 1) = -g_cav * phi;
}

// ================================================================
// k-grid for 1D chain, 2-site unit cell
//
// h(k) = -t*(1+cos k)*tau_x - t*sin(k)*tau_y + (eps_perp - mu)*I
//      = [ eps_perp - mu,       -t*(1 + e^{-ik}) ]
//        [ -t*(1 + e^{ik}),     eps_perp - mu    ]
//
// Uniform weights (trapezoidal rule, optimal for periodic integrands).
//
// Reduced to k in [0,pi]: h(2*pi-k) = h(k)^T exactly (off-diagonal entries
// swap), so G_{2*pi-k} = G_k^T. Every quantity this code reads off Gloc
// (individual diagonal elements n0/n1, and Im-tr-based ratios in
// extract_F_binned) is invariant under transpose, so visiting only k in
// [0,pi] with the interior points double-weighted (they represent two
// physical k, k and 2*pi-k) reproduces those exactly, at roughly half the
// per-(k,w) matrix work. compute_gloc_ksum additionally symmetrizes
// (G_k+G_k^T)/2 before accumulating, so even Gloc's off-diagonal elements
// come out exactly right (at no extra inversion cost -- just a transpose).
// The two endpoints k=0 and k=pi (when nk is even) are their own mirror
// image (2*pi-0=0, 2*pi-pi=pi) and keep single weight, not doubled.
//
// nk keeps its original meaning (sets the k-grid spacing dk=2*pi/nk, same
// physical resolution as before); the number of points actually stored
// (hk.size()) is roughly nk/2+1, not nk -- callers should re-read nk as
// hk.size() after calling this.
//
// TODO for 2D: add an outer loop over eps_perp with a transverse DOS,
// e.g. semicircular for Bethe lattice or explicit k_perp summation.
// ================================================================

void setup_kgrid(int nk, double t_hop, double mu, double eps_perp,
                 vector<cdmatrix> &hk, vector<double> &wk) {
    hk.clear();
    wk.clear();
    cplx ii(0.0, 1.0);
    double dk = 2.0 * M_PI / nk;
    int n_half = nk / 2; // walk i = 0..n_half, i.e. k = 0..(at most) pi

    for (int i = 0; i <= n_half; i++) {
        double k = i * dk;
        bool self_paired = (i == 0) || (nk % 2 == 0 && i == n_half);
        double weight = self_paired ? (1.0 / nk) : (2.0 / nk);

        cdmatrix h(2, 2);
        cplx off_diag = -t_hop * (1.0 + exp(-ii * k));
        h(0, 0) = eps_perp - mu;
        h(0, 1) = off_diag;
        h(1, 0) = conj(off_diag);
        h(1, 1) = eps_perp - mu;

        hk.push_back(h);
        wk.push_back(weight);
    }
}

// ================================================================
// G_loc via explicit k-summation
//
// G_k^R(w) = (w*I - h_k - Hartree - Sigma_hub^R(w) - Sigma_cav^R(k,w))^{-1}
// G_k^<(w) = G_k^R * (Sigma_hub^<(w) + Sigma_cav^<(k,w)) * G_k^A
// G_loc(w) = sum_k w_k * G_k(w)
//
// Sigma_hub is local (k-independent, from FFT convolution on Gloc, as
// before). Sigma_cav is k-dependent and built analytically at each k from
// the quasiparticle bands/projectors and the distribution function F_k
// (see eval_Sigma_cav_k above) -- no FFT and no full k-resolved GF storage
// needed for the cavity part.
//
// hk/wk only cover k in [0,pi] (see setup_kgrid): each G_k is symmetrized,
// (G_k+G_k^T)/2, before being weighted and accumulated. This costs nothing
// extra (no second inversion, just a transpose) and makes the sum exactly
// equal to summing over the full k in [0,2*pi) (see setup_kgrid for why).
// ================================================================

void compute_gloc_ksum(int nk, const vector<cdmatrix> &hk,
                       const vector<double> &wk,
                       const cdmatrix &Hartree, const GF &Sigma_hub,
                       const vector<BandData> &bands,
                       const vector<array<double, 2>> &F_k,
                       double g_eff_sq, double delta_cav, double Gamma_cav,
                       GF &Gloc, double beta_thermal = -1.0) {
    Gloc.clear();
    cdmatrix I2 = cdmatrix::Identity(2, 2);
    vector<array<cdmatrix, 2>> TPT = compute_TPT(bands);

#pragma omp parallel for schedule(dynamic)
    for (long w = 0; w < Gloc.ngrid_; w++) {
        double omega = Gloc.grid_[w];
        cdmatrix Gret_sum = cdmatrix::Zero(2, 2);
        cdmatrix Gles_sum = cdmatrix::Zero(2, 2);
        cdmatrix SigHubR = Sigma_hub.Retarded[w];
        cdmatrix SigHubL = Sigma_hub.Lesser[w];

        for (int k = 0; k < nk; k++) {
            cdmatrix SigCavR, SigCavL;
            eval_Sigma_cav_k(omega, bands[k].eps, TPT[k], F_k[k],
                             g_eff_sq, delta_cav, Gamma_cav, SigCavR, SigCavL,
                             beta_thermal);

            cdmatrix SigR = SigHubR + SigCavR;
            cdmatrix SigL = SigHubL + SigCavL;

            cdmatrix GkR = (omega * I2 - hk[k] - Hartree - SigR).inverse();
            cdmatrix GkL = GkR * SigL * GkR.adjoint();
            // NOTE: GkR.transpose() (an expression referencing GkR's own storage)
            // must be evaluated into a plain temporary before assigning back into
            // GkR -- "GkR = 0.5*(GkR+GkR.transpose())" is Eigen's classic aliasing
            // trap (mat = mat.transpose() corrupts off-diagonal entries) and was a
            // real bug here (diagonal elements happened to survive since transpose
            // doesn't move them, which is exactly why it went unnoticed at first).
            cdmatrix GkR_T = GkR.transpose();
            cdmatrix GkL_T = GkL.transpose();
            GkR = 0.5 * (GkR + GkR_T);
            GkL = 0.5 * (GkL + GkL_T);
            Gret_sum += wk[k] * GkR;
            Gles_sum += wk[k] * GkL;
        }

        Gloc.Retarded[w] = Gret_sum;
        Gloc.Lesser[w] = Gles_sum;
    }
}

// Diagnostic only: k-averaged cavity self-energy, for output purposes
// (Sigma_cav is now genuinely k-dependent; this is sum_k wk[k]*Sigma_cav(k,w)).
void compute_Sigma_cav_kavg(int nk, const vector<double> &wk,
                            const vector<BandData> &bands,
                            const vector<array<double, 2>> &F_k,
                            double g_eff_sq, double delta_cav, double Gamma_cav,
                            GF &Sigma_cav_avg) {
    Sigma_cav_avg.clear();
    vector<array<cdmatrix, 2>> TPT = compute_TPT(bands);

#pragma omp parallel for schedule(dynamic)
    for (long w = 0; w < Sigma_cav_avg.ngrid_; w++) {
        double omega = Sigma_cav_avg.grid_[w];
        cdmatrix Rsum = cdmatrix::Zero(2, 2), Lsum = cdmatrix::Zero(2, 2);
        for (int k = 0; k < nk; k++) {
            cdmatrix SigR, SigL;
            eval_Sigma_cav_k(omega, bands[k].eps, TPT[k], F_k[k],
                             g_eff_sq, delta_cav, Gamma_cav, SigR, SigL);
            Rsum += wk[k] * SigR;
            Lsum += wk[k] * SigL;
        }
        Sigma_cav_avg.Retarded[w] = Rsum;
        Sigma_cav_avg.Lesser[w] = Lsum;
    }
}

// ================================================================
// Output: full 2x2 GF (all matrix elements)
// ================================================================

void outputGF_full(GF &G, string filename) {
    ofstream out(filename);
    out << setprecision(15);
    out << "# omega";
    for (int i = 0; i < G.size1_; i++)
        for (int j = 0; j < G.size2_; j++)
            out << "\tRe(GR_" << i << j << ")\tIm(GR_" << i << j << ")";
    for (int i = 0; i < G.size1_; i++)
        for (int j = 0; j < G.size2_; j++)
            out << "\tRe(GL_" << i << j << ")\tIm(GL_" << i << j << ")";
    out << "\n";

    for (long w = 0; w < G.ngrid_; w++) {
        out << G.grid_[w];
        for (int i = 0; i < G.size1_; i++)
            for (int j = 0; j < G.size2_; j++)
                out << "\t" << G.Retarded[w](i, j).real()
                    << "\t" << G.Retarded[w](i, j).imag();
        for (int i = 0; i < G.size1_; i++)
            for (int j = 0; j < G.size2_; j++)
                out << "\t" << G.Lesser[w](i, j).real()
                    << "\t" << G.Lesser[w](i, j).imag();
        out << "\n";
    }
    out.close();
}

// ================================================================
// Thermal-FDT regression test
//
// Force the cavity photon propagator to be EXACTLY thermal at a chosen
// inverse temperature beta_test (D^R kept as-is, D^K replaced by its exact
// FDT value via eval_D_thermal). By the fluctuation-dissipation theorem,
// fermions coupled to a genuinely thermal bath MUST reach exactly that same
// temperature at self-consistency, regardless of g_eff_sq -- this is a hard
// mathematical fact, not an approximation. So after running the real
// self-consistency loop (same compute_gloc_ksum/extract_F_binned used in
// production) with this thermal D, the converged F(eps) must equal
// tanh(beta_test*eps/2) to within binning
// resolution. Any larger deviation is a genuine bug in Sigma_cav (sign,
// missing factor, wrong index, ...), not a physics/approximation effect.
//
// Self-contained (no parameter file, fixed test parameters below) so it
// stays a fixed, repeatable check across future code changes:
//   ./quantum_boltz_cavity.ex --thermal-test
//   ./quantum_boltz_cavity.ex --thermal-test <param_file>   (override defaults below)
//
// <param_file> uses the same __key=value format as normal run parameters.
// Every key below is optional; anything not present keeps its default.
// __beta_test, if given (and >= 0), fixes the test temperature directly
// instead of the default T_LF derived from delta_cav/Gamma_cav (T_LF is what
// the fermions should thermalize to under FDT if D^K were the *real*
// driven-dissipative one -- __beta_test lets thermal_test.py test the FDT
// check itself at any chosen temperature, independent of T_LF).
// Recognized keys: __nk __numOfFreq __freq_cutoff __t_hop __delta_cav
// __Gamma_cav __g_eff_sq __Niter __mixing __err __Jbath __beta_test
// ================================================================

void run_thermal_fdt_test(const string &param_file) {
    // ---- Default test parameters, optionally overridden from param_file ----
    int nk = 200;
    int numOfFreq_i = 4000;
    double freq_cutoff = 20.0;
    double t_hop = 0.5, mu = 0.0, eps_perp = 0.0;
    double delta_cav = 2.0, Gamma_cav = 2.0;
    double g_eff_sq = 0.05;
    int Niter = 80;
    double mixing = 0.5, lerr = 1e-11;
    double Jbath = 0.05;
    double beta_test_override = -1.0; // < 0 means "use T_LF"

    if (!param_file.empty()) {
        find_param(param_file, "__nk", nk);
        find_param(param_file, "__numOfFreq", numOfFreq_i);
        find_param(param_file, "__freq_cutoff", freq_cutoff);
        find_param(param_file, "__t_hop", t_hop);
        find_param(param_file, "__delta_cav", delta_cav);
        find_param(param_file, "__Gamma_cav", Gamma_cav);
        find_param(param_file, "__g_eff_sq", g_eff_sq);
        find_param(param_file, "__Niter", Niter);
        find_param(param_file, "__mixing", mixing);
        find_param(param_file, "__err", lerr);
        find_param(param_file, "__Jbath", Jbath);
        find_param(param_file, "__beta_test", beta_test_override);
    }

    long numOfFreq = numOfFreq_i;
    double grid_spacing = freq_cutoff * 2.0 / numOfFreq;
    double beta_test = (beta_test_override >= 0.0)
        ? beta_test_override
        : 4.0 * delta_cav / (delta_cav * delta_cav + 0.5 * Gamma_cav * Gamma_cav); // = beta_LF
    double tol = 0.05; // generous vs. nk/numOfFreq binning resolution, tight vs. a real sign/factor bug
    int size = 2;

    cout << "=== Thermal-FDT regression test ===" << endl;
    if (!param_file.empty()) cout << "(parameters from " << param_file << ")" << endl;
    cout << "nk=" << nk << " numOfFreq=" << numOfFreq << " freq_cutoff=" << freq_cutoff
         << " t_hop=" << t_hop << endl;
    cout << "delta_cav=" << delta_cav << " Gamma_cav=" << Gamma_cav << " g_eff_sq=" << g_eff_sq
         << " Niter=" << Niter << " mixing=" << mixing << " err=" << lerr << " Jbath=" << Jbath << endl;
    cout << "beta_test = " << beta_test << " (T_test = " << 1.0 / beta_test << ")"
         << (beta_test_override >= 0.0 ? " [explicit __beta_test]" : " [= T_LF]") << endl;

    vector<cdmatrix> hk;
    vector<double> wk;
    setup_kgrid(nk, t_hop, mu, eps_perp, hk, wk);
    nk = (int)hk.size(); // setup_kgrid reduces to k in [0,pi]; see its comment

    GF Gloc(grid_spacing, numOfFreq, size);
    GF Sigma_hub(grid_spacing, numOfFreq, size); // stays zero: no Hubbard in this test
    GF Bath(grid_spacing, numOfFreq, size);
    cdmatrix Hartree = cdmatrix::Zero(2, 2); // no Hartree: isolates the cavity Sigma_cav FDT check

    setBath_diag(Jbath, 1.0, Bath); // seed only; its temperature doesn't matter here

    vector<BandData> bands;
    compute_bands_k(hk, Hartree, bands);
    vector<array<double, 2>> F_k(nk, {0.5, 0.5});
    compute_gloc_ksum(nk, hk, wk, Hartree, Bath, bands, F_k, 0.0, delta_cav, Gamma_cav, Gloc);

    // ---- Fixed-point check: feed the EXACT F=tanh(beta_test*eps/2) in, do a
    // SINGLE pass, and see if it comes back out (near-)unchanged. FDT
    // guarantees this must be an exact fixed point for any g_eff_sq > 0; if a
    // single pass already moves far from it, the bug is in the formula
    // itself, not iteration dynamics/convergence.
    {
        vector<array<double, 2>> F_exact(nk);
        for (int k = 0; k < nk; k++)
            for (int nu = 0; nu < 2; nu++)
                F_exact[k][nu] = tanh(beta_test * bands[k].eps[nu] / 2.0);

        GF Gfp(grid_spacing, numOfFreq, size);
        compute_gloc_ksum(nk, hk, wk, Hartree, Sigma_hub, bands, F_exact,
                          g_eff_sq, delta_cav, Gamma_cav, Gfp, beta_test);
        vector<array<double, 2>> F_after;
        extract_F_binned(Gfp, bands, F_after);

        double fp_max_err = 0.0;
        for (int k = 0; k < nk; k++)
            for (int nu = 0; nu < 2; nu++)
                fp_max_err = max(fp_max_err, abs(F_after[k][nu] - F_exact[k][nu]));
        cout << "[FIXED-POINT CHECK] max |F_after_one_pass - F_exact| = " << fp_max_err
             << " (should be ~0 if F_exact is a fixed point)" << endl;
    }

    // ---- Stability check: perturb F_exact by a known amount, do ONE pass,
    // and see if the perturbation grows or shrinks. If F_exact is a stable
    // fixed point, perturbation should shrink (or at least not grow); if it
    // grows, that explains why iteration runs away from it toward saturation
    // even though F_exact itself is close to self-consistent.
    {
        vector<array<double, 2>> F_pert(nk);
        double pert_amp = 0.1;
        for (int k = 0; k < nk; k++)
            for (int nu = 0; nu < 2; nu++) {
                double exact = tanh(beta_test * bands[k].eps[nu] / 2.0);
                double sign = (nu == 1) ? 1.0 : -1.0; // keep it odd, consistent with chiral symmetry
                F_pert[k][nu] = max(-1.0, min(1.0, exact + sign * pert_amp));
            }

        GF Gpert(grid_spacing, numOfFreq, size);
        compute_gloc_ksum(nk, hk, wk, Hartree, Sigma_hub, bands, F_pert,
                          g_eff_sq, delta_cav, Gamma_cav, Gpert, beta_test);
        vector<array<double, 2>> F_pert_after;
        extract_F_binned(Gpert, bands, F_pert_after);

        double in_dev = 0.0, out_dev = 0.0;
        for (int k = 0; k < nk; k++)
            for (int nu = 0; nu < 2; nu++) {
                double exact = tanh(beta_test * bands[k].eps[nu] / 2.0);
                in_dev = max(in_dev, abs(F_pert[k][nu] - exact));
                out_dev = max(out_dev, abs(F_pert_after[k][nu] - exact));
            }
        cout << "[STABILITY CHECK] perturbation from exact: in=" << in_dev
             << " -> after one pass=" << out_dev
             << (out_dev > in_dev ? "  (GROWING -> unstable)" : "  (shrinking -> stable)") << endl;
    }

    double GF_err = 10.0;
    int iter = 0;
    while (iter < Niter && GF_err > lerr) {
        GF Gold = Gloc;

        extract_F_binned(Gloc, bands, F_k);

        compute_gloc_ksum(nk, hk, wk, Hartree, Sigma_hub, bands, F_k,
                          g_eff_sq, delta_cav, Gamma_cav, Gloc, beta_test);

        mixing_G(Gloc, Gold, mixing);
        GF_err = GF2norm(Gloc, Gold);
        iter++;
    }
    cout << "Converged after " << iter << " iterations, err=" << GF_err << endl;

    vector<array<double, 2>> F_final;
    extract_F_binned(Gloc, bands, F_final); // raw extraction, not re-symmetrized:
                                             // oddness should emerge on its own if correct

    double max_err = 0.0;
    int worst_k = -1, worst_nu = -1;
    vector<pair<double, pair<double, double>>> plot_data; // eps -> (F_numeric, F_exact)
    for (int k = 0; k < nk; k++) {
        for (int nu = 0; nu < 2; nu++) {
            double eps = bands[k].eps[nu];
            double expected = tanh(beta_test * eps / 2.0);
            double err = abs(F_final[k][nu] - expected);
            if (err > max_err) { max_err = err; worst_k = k; worst_nu = nu; }
            plot_data.push_back({eps, {F_final[k][nu], expected}});
        }
    }

    sort(plot_data.begin(), plot_data.end(),
        [](const auto &a, const auto &b) { return a.first < b.first; });
    string data_file = "thermal_fdt_test.dat";
    ofstream out(data_file);
    out << setprecision(10);
    out << "# eps\tF_numeric\tF_exact_thermal(tanh(beta_test*eps/2))\n";
    for (auto &row : plot_data)
        out << row.first << "\t" << row.second.first << "\t" << row.second.second << "\n";
    out.close();
    cout << "Wrote " << data_file << " (plot with thermal_test.py)" << endl;

    cdmatrix rho(2, 2);
    DensityMatrix(Gloc, rho);
    cout << "n0 = " << rho(0, 0).real() << ", n1 = " << rho(1, 1).real() << " (should both be 0.5)" << endl;
    cout << "Max |F_numeric - F_exact_thermal| = " << max_err << " at k=" << worst_k
         << " nu=" << worst_nu << " (eps=" << bands[worst_k].eps[worst_nu] << ")" << endl;

    if (max_err < tol) {
        cout << "[THERMAL-FDT TEST] PASS (tolerance " << tol << ")" << endl;
    } else {
        cout << "[THERMAL-FDT TEST] FAIL (tolerance " << tol << ") -- fermions did NOT "
             << "thermalize to the photon's temperature. Check Sigma_cav's Lesser/Retarded "
             << "sign, normalization, and nu<->TPT index pairing." << endl;
    }
}

// ================================================================
// MAIN
// ================================================================

int main(int argc, char **argv) {
    if (argc > 1 && string(argv[1]) == string("--thermal-test")) {
        string thermal_param_file = (argc > 2) ? string(argv[2]) : string("");
        run_thermal_fdt_test(thermal_param_file);
        return 0;
    }

    // ---- Parameters ----
    double freq_cutoff, grid_spacing;
    long numOfFreq;
    int nk;
    double t_hop, mu, eps_perp;
    double U;
    double delta_cav, Gamma_cav, g_cav, g_eff_sq;
    double Jbath, beta;
    int Niter;
    double mixing, lerr;
    int tmp_int;

    if (argv[1] == NULL) {
        cout << "Usage: ./quantum_boltz_cavity.ex <param_file>" << endl;
        return 1;
    }

    find_param(argv[1], "__numOfFreq", tmp_int);
    numOfFreq = tmp_int;
    find_param(argv[1], "__freq_cutoff", freq_cutoff);
    find_param(argv[1], "__nk", nk);
    find_param(argv[1], "__t_hop", t_hop);
    find_param(argv[1], "__mu", mu);
    find_param(argv[1], "__eps_perp", eps_perp);
    find_param(argv[1], "__U", U);
    find_param(argv[1], "__delta_cav", delta_cav);
    find_param(argv[1], "__Gamma_cav", Gamma_cav);
    find_param(argv[1], "__g_cav", g_cav);
    find_param(argv[1], "__g_eff_sq", g_eff_sq);
    find_param(argv[1], "__Jbath", Jbath);
    find_param(argv[1], "__beta", beta);
    find_param(argv[1], "__Niter", Niter);
    find_param(argv[1], "__mixing", mixing);
    find_param(argv[1], "__err", lerr);

    grid_spacing = freq_cutoff * 2.0 / numOfFreq;

    cout << "=== Parameters ===" << endl;
    cout << "numOfFreq = " << numOfFreq << ", freq_cutoff = " << freq_cutoff
         << ", grid_spacing = " << grid_spacing << endl;
    cout << "nk = " << nk << endl;
    cout << "t_hop = " << t_hop << ", mu = " << mu << ", eps_perp = " << eps_perp << endl;
    cout << "U = " << U << endl;
    cout << "delta_cav = " << delta_cav << ", Gamma_cav = " << Gamma_cav
         << ", g_cav = " << g_cav << ", g_eff_sq = " << g_eff_sq << endl;
    cout << "Jbath = " << Jbath << ", beta = " << beta << endl;
    cout << "Niter = " << Niter << ", mixing = " << mixing
         << ", err = " << lerr << endl;
    cout << "==================" << endl;

    // ---- Setup ----
    int size = 2;

    vector<cdmatrix> hk;
    vector<double> wk;
    setup_kgrid(nk, t_hop, mu, eps_perp, hk, wk);
    nk = (int)hk.size(); // setup_kgrid reduces to k in [0,pi]; see its comment
    cout << "nk (actual grid points, k in [0,pi]) = " << nk << endl;

    GF Gloc(grid_spacing, numOfFreq, size);
    GF Sigma_hub(grid_spacing, numOfFreq, size);
    GF Sigma_cav(grid_spacing, numOfFreq, size); // diagnostic only: k-averaged cavity self-energy
    GF Sigma_tot(grid_spacing, numOfFreq, size); // diagnostic only: k-averaged total self-energy
    GF Bath(grid_spacing, numOfFreq, size);
    GF Dcav(grid_spacing, numOfFreq, 1); // 1x1 scalar cavity propagator, diagnostic output only
    cdmatrix Hartree = cdmatrix::Zero(2, 2);
    vector<BandData> bands;
    vector<array<double, 2>> F_k;

    fft_solver solver(numOfFreq, FFTW_ESTIMATE);

    // Initialize cavity propagator (analytical, on frequency grid; used only for Dcav.out,
    // the self-energy itself is evaluated in closed form via eval_D/eval_Sigma_cav_k)
    set_cavity_propagator(delta_cav, Gamma_cav, Dcav);

    // Initialize Markovian bath
    double beta_LF = 4* delta_cav /( delta_cav* delta_cav + 0.5 * Gamma_cav * Gamma_cav );
    setBath_diag(Jbath, beta_LF, Bath); // beta is just initial temperature

    // ---- Seed: non-interacting G_loc (bath only, no Hubbard, no cavity) ----
    compute_bands_k(hk, Hartree, bands);
    F_k.assign(nk, {0.5, 0.5});
    compute_gloc_ksum(nk, hk, wk, Hartree, Bath, bands, F_k, 0.0, delta_cav, Gamma_cav, Gloc);

    cdmatrix rho(2, 2);
    DensityMatrix(Gloc, rho);
    cout << "Initial (non-interacting): n0 = " << rho(0, 0).real()
         << ", n1 = " << rho(1, 1).real() << endl;

    // ---- Self-consistency loop ----
    double GF_err = 10.0;
    int iter = 0;

    while (iter < Niter && GF_err > lerr) {
        cout << "------------ Iteration " << iter << " -------------" << endl;

        GF Gold = Gloc;

        // 1. Hartree from current G_loc
        compute_hartree(g_cav, delta_cav, Gamma_cav, Gloc, Hartree);

        // 2. Hubbard self-energy from current G_loc (bold perturbation theory, local/k-independent)
        //    TODO: for IPT, replace Gloc by Gweiss = (G_loc^{-1} + Sigma_hub)^{-1}
        //    in the Hubbard self-energy computation.
        compute_Sigma_hubbard(U, Gloc, Sigma_hub, solver);

        // 3. Quasiparticle bands/projectors (from current Hartree) and distribution
        //    function F(eps_{k,nu}) (from current G_loc), feeding the analytic,
        //    k-dependent cavity self-energy evaluated inside compute_gloc_ksum.
        compute_bands_k(hk, Hartree, bands);
        extract_F_binned(Gloc, bands, F_k);
        // extract_F_direct(Gloc, bands, F_k); // pointwise/interpolated alternative,
        // valid once nk is large enough that Im tr G^R is smooth on the omega grid
        // (empirically not yet the case at nk=200 -- see conversation).

        // 4. New G_loc from k-summation (Hubbard local + cavity k-dependent)
        compute_gloc_ksum(nk, hk, wk, Hartree, Sigma_hub, bands, F_k,
                          g_eff_sq, delta_cav, Gamma_cav, Gloc);

        // 5. Mixing for convergence
        mixing_G(Gloc, Gold, mixing);

        // 6. Convergence check
        GF_err = GF2norm(Gloc, Gold);
        DensityMatrix(Gloc, rho);
        cout << "  err = " << GF_err
             << "  n0 = " << rho(0, 0).real()
             << "  n1 = " << rho(1, 1).real() << endl;

        iter++;
    }

    // ---- Output converged results ----
    compute_hartree(g_cav, delta_cav, Gamma_cav, Gloc, Hartree);
    DensityMatrix(Gloc, rho);

    // Diagnostic-only k-averaged cavity/total self-energy, from the same
    // bands/F_k used in the last self-consistency iteration above.
    compute_Sigma_cav_kavg(nk, wk, bands, F_k, g_eff_sq, delta_cav, Gamma_cav, Sigma_cav);
    Sigma_tot.clear();
    Sigma_tot.incr(Sigma_hub);
    Sigma_tot.incr(Sigma_cav);

    cout << "\n=== Converged after " << iter << " iterations ===" << endl;
    cout << "n0 = " << rho(0, 0).real() << ", n1 = " << rho(1, 1).real() << endl;
    cout << "Hartree(0,0) = " << Hartree(0, 0).real()
         << ", Hartree(1,1) = " << Hartree(1, 1).real() << endl;

    cout << "beta_LF = " << beta_LF << " (T_LF = " << 1.0 / beta_LF << ")" << endl;

    // Sanity check only: extract_F_binned already clamps F(eps_{k,nu}) to
    // [-1,1], so this should never fire -- kept as a safety net in case that
    // changes.
    for (int k = 0; k < nk; k++) {
        for (int nu = 0; nu < 2; nu++) {
            if (abs(F_k[k][nu]) > 1.0) {
                cout << "  WARNING: F(eps=" << bands[k].eps[nu] << ") = " << F_k[k][nu]
                     << " outside [-1,1] at k=" << k << " nu=" << nu << endl;
            }
        }
    }

    // Spectral sum rule check: integral dw/(2*pi) * (-2*Im(G^R_{aa})) should be 1
    double sumrule0 = 0.0, sumrule1 = 0.0;
    for (long w = 0; w < Gloc.ngrid_; w++) {
        sumrule0 += -2.0 * Gloc.Retarded[w](0, 0).imag();
        sumrule1 += -2.0 * Gloc.Retarded[w](1, 1).imag();
    }
    sumrule0 *= grid_spacing / (2.0 * M_PI);
    sumrule1 *= grid_spacing / (2.0 * M_PI);
    cout << "Spectral sum rule: A0 = " << sumrule0 << ", A1 = " << sumrule1
         << " (should be 1)" << endl;

    outputGF_full(Gloc, "Gloc_converged.out");
    outputGF_full(Sigma_hub, "Sigma_hub_converged.out");
    outputGF_full(Sigma_cav, "Sigma_cav_converged.out");
    outputGF_full(Sigma_tot, "Sigma_tot_converged.out");
    outputGF_full(Dcav, "Dcav.out");

    fftw_cleanup();
    return 0;
}
