#ifndef ness_fft_weight
#define ness_fft_weight
//#include "ness.hpp"
//#include "ness_fft_decl.hpp"
#include "ness_GF_decl.hpp"
namespace ness
{
class fft_solver_weight
{
	public:
	fft_solver_weight(int nfreq, size_t FFTW_FLAG) : nfreq_(nfreq)
	{
		Nft_ = 3 * nfreq_;
		in = new fftw_complex[Nft_];
		out = new fftw_complex[Nft_];
		p_for = fftw_plan_dft_1d(Nft_, in, out, FFTW_FORWARD, FFTW_FLAG);
		p_back = fftw_plan_dft_1d(Nft_, in, out, FFTW_BACKWARD, FFTW_FLAG);
	}

	~fft_solver_weight()
	{
		delete[] in;
		delete[] out;
	}
	int to_time(GF &outG, const GF &inG, int set_dt = 0);
	int to_freq(GF &outG, const GF &inG, int set_dw = 0);

	fftw_complex* in;
	fftw_complex* out;
	int Nft_, nfreq_;
	fftw_plan p_for, p_back;

};
}
#endif
