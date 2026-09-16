#ifndef ness_fft
#define ness_fft
//#include "ness.hpp"
//#include "ness_fft_decl.hpp"
#include "ness_GF_decl.hpp"
namespace ness
{

class fft_solver
{
	public:
	fft_solver(int nfreq, size_t FFTW_FLAG) : nfreq_(nfreq)
	{
		Nft_ = 3 * nfreq_;
		in = new fftw_complex[Nft_];
		out = new fftw_complex[Nft_];
		p_for = fftw_plan_dft_1d(Nft_, in, out, FFTW_FORWARD, FFTW_FLAG);
		p_back = fftw_plan_dft_1d(Nft_, in, out, FFTW_BACKWARD, FFTW_FLAG);
        corfac_exists=false;
	}

	~fft_solver()
	{
		delete[] in;
		delete[] out;
        if(corfac_exists) delete[] corfac;
	}
	int to_time(GF &outG, const GF &inG, int set_dt = 0);
	int to_freq(GF &outG, const GF &inG, int set_dw = 0);
	//int pp_to_time(GF &outG, const GF &inG, int set_dt = 0);
	//nt pp_to_freq(GF &outG, const GF &inG, int set_dw = 0);
	//nt lpp_to_time(GF &outG, const GF &inG, int set_dt = 0);
	//nt lpp_to_freq(GF &outG, const GF &inG, int set_dw = 0);

	
    void compute_corfac();
    int     to_time_corfac(GF &outG, const GF &inG);
    
    double *corfac;
	fftw_complex* in;
	fftw_complex* out;
	int Nft_, nfreq_;
	fftw_plan p_for, p_back;
    bool corfac_exists;

};
}
#endif
