#ifndef FFT_IMPL
#define FFT_IMPL

#ifndef STAND_ALONE
	#include "cntr.hpp"
#endif
//#include "ness_GF_decl.hpp"
#include "ness_fft_decl.hpp"

//Jiajun Li, 2018 June

namespace ness {
#ifdef TODO_IS_DONE
int fft_solver::lpp_to_freq(GF &outG, const GF &inG, int set_dw)
{
	#ifndef STAND_ALONE
	CNTR_ASSERT_EQ(NESS_ASSERT_0, inG.size1_ , outG.size1_, __PRETTY_FUNCTION__);
	CNTR_ASSERT_EQ(NESS_ASSERT_0, inG.size2_ , outG.size2_, __PRETTY_FUNCTION__);
	CNTR_ASSERT_EQ(NESS_ASSERT_0, outG.ngrid_ , this -> Nft_, __PRETTY_FUNCTION__);
	CNTR_ASSERT_EQ(NESS_ASSERT_0, inG.ngrid_ , this -> Nft_ / 2 + 1, __PRETTY_FUNCTION__);
	CNTR_ASSERT_EQ(NESS_ASSERT_0, inG.gf_type_ , time_gf, __PRETTY_FUNCTION__);
	CNTR_ASSERT_EQ(NESS_ASSERT_0, outG.gf_type_ , freq_gf, __PRETTY_FUNCTION__);
	#else
	assert(inG.size1_ == outG.size1_);
	assert(inG.size2_ == outG.size2_);
	assert(outG.ngrid_ == this -> Nft_);
	assert(inG.ngrid_ == this -> Nft_ / 2 + 1);
	assert(inG.gf_type_ == time_gf);
	assert(outG.gf_type_ == freq_gf);
	#endif
	for(int idx1 = 0; idx1 < inG.size1_; idx1++)
	{
		for(int idx2 = 0; idx2 < inG.size2_; idx2++)
		{
			//compute gret
			//std::memset(in, 0, sizeof(fftw_complex) * Nft_);
			for(int w = 0; w < Nft_ / 2; w++)
			{
				in[w][0] = (*(inG.p_ret(w, idx1, idx2))).real();
				in[w][1] = (*inG.p_ret(w, idx1, idx2)).imag();
				//in[Nft_ - w - 1][0] = -(*inG.p_ret(w + 1, idx2, idx1)).real();
				//in[Nft_ - w - 1][1] = (*inG.p_ret(w + 1, idx2, idx1)).imag();
			}
			for(int w = Nft_ / 2; w < Nft_; w++) {in[w][0] = 0.0; in[w][1] = 0.0;}

			fftw_execute(p_back);
			for(int w = 0; w < Nft_; w++)
			{
				(*outG.p_ret(w, idx1, idx2)) = out[w][0] + out[w][1] * II;
				//(*outG.p_ret(nfreq_ - w - 1, idx1, idx2)) = out[Nft_ - w - 1][0] + out[Nft_ - w - 1][1] * II;
			}

			//compute glss
			for(int w = 0; w < Nft_ / 2; w++)
			{
				in[w][0] = (*inG.p_les(w, idx1, idx2)).real();
				in[w][1] = (*inG.p_les(w, idx1, idx2)).imag();
				in[Nft_ - w - 1][0] = -(*inG.p_les(w + 1, idx2, idx1)).real();
				in[Nft_ - w - 1][1] = (*inG.p_les(w + 1, idx2, idx1)).imag();
			}
			in[Nft_ / 2][0] = (*inG.p_les(Nft_ / 2, idx2, idx1)).real();
			in[Nft_ / 2][1] = (*inG.p_les(Nft_ / 2, idx2, idx1)).imag();

			fftw_execute(p_back);
			for(int w = 0; w < Nft_; w++)
			{
				(*outG.p_les(w, idx1, idx2)) = out[w][0] + out[w][1] * II;
				//(*outG.p_les(nfreq_ - w - 1, idx1, idx2)) = out[Nft_ - w - 1][0] + out[Nft_ - w - 1][1] * II;
			}


		}
	}
	outG.smul(inG.dgrid_);
	if(set_dw) outG.reset_grid(2.0 * M_PI / (inG.dgrid_ * (Nft_ - 1)));
}


int fft_solver::lpp_to_time(GF &outG, const GF &inG, int set_dt)
{

	#ifndef STAND_ALONE
	CNTR_ASSERT_EQ(NESS_ASSERT_0, inG.size1_ , outG.size1_, __PRETTY_FUNCTION__);
	CNTR_ASSERT_EQ(NESS_ASSERT_0, inG.size2_ , outG.size2_, __PRETTY_FUNCTION__);
	CNTR_ASSERT_EQ(NESS_ASSERT_0, inG.ngrid_ , this -> Nft_, __PRETTY_FUNCTION__);
	CNTR_ASSERT_EQ(NESS_ASSERT_0, outG.ngrid_ , this -> Nft_ / 2 + 1, __PRETTY_FUNCTION__);
	CNTR_ASSERT_EQ(NESS_ASSERT_0, outG.gf_type_ , time_gf, __PRETTY_FUNCTION__);
	CNTR_ASSERT_EQ(NESS_ASSERT_0, inG.gf_type_ , freq_gf, __PRETTY_FUNCTION__);
	#else
	assert(inG.size1_ == outG.size1_);
	assert(inG.size2_ == outG.size2_);
	assert(inG.ngrid_ == this -> Nft_);
	assert(outG.ngrid_ == this -> Nft_ / 2 + 1);
	assert(outG.gf_type_ == time_gf);
	assert(inG.gf_type_ == freq_gf);
	#endif

	
	for(int idx1 = 0; idx1 < inG.size1_; idx1++)
	{
		for(int idx2 = 0; idx2 < inG.size2_; idx2++)
		{
			//compute gret
			//for(int w = 0; w < Nft_; w++) {in[w][0] = 0.0; in[w][1] = 0.0;}
			for(int w = 0; w < Nft_; w++)
			{
				in[w][0] = ((*inG.p_ret(w, idx1, idx2)).real() - (*inG.p_ret(w, idx2, idx1)).real());
				in[w][1] = ((*inG.p_ret(w, idx1, idx2)).imag() + (*inG.p_ret(w, idx2, idx1)).imag());
				//in[Nft_ - w - 1][0] = ((*inG.p_ret(nfreq_ - w - 1, idx1, idx2)).real() - (*inG.p_ret(nfreq_ - w - 1, idx2, idx1)).real());
				//in[Nft_ - w - 1][1] = ((*inG.p_ret(nfreq_ - w - 1, idx1, idx2)).imag() + (*inG.p_ret(nfreq_ - w - 1, idx2, idx1)).imag());

			}
			fftw_execute(p_for);
			for(int w = 0; w < Nft_ / 2; w++)
			{
				(*outG.p_ret(w, idx1, idx2)) = out[w][0] + II * out[w][1];
			}
			
			//for(int w = Nft_ / 2; w < Nft_; w++) {in[w][0] = 0.0; in[w][1] = 0.0;}

			for(int w = 0; w < Nft_; w++)
			{
				in[w][0] = (*inG.p_les(w, idx1, idx2)).real();
				in[w][1] = (*inG.p_les(w, idx1, idx2)).imag();
				//in[Nft_ - w - 1][0] = (*inG.p_les(nfreq_ - w - 1, idx1, idx2)).real();
				//in[Nft_ - w - 1][1] = (*inG.p_les(nfreq_ - w - 1, idx1, idx2)).imag();
			}
			fftw_execute(p_for);
			for(int w = 0; w < Nft_ / 2 + 1; w++)
			{
				(*outG.p_les(w, idx1, idx2)) = out[w][0] + out[w][1] * II;
			}
			(*outG.p_ret(0, idx1, idx2)) *= 0.5;
		}
	}
	outG.smul(inG.dgrid_ / (2.0 * M_PI));
	if(set_dt) outG.reset_grid(2.0 * M_PI / (inG.dgrid_ * (Nft_ - 1)));
}


//pp_versions have G^r(t) = G^<(t) (t > 0) and 2 * ImG^r(w) = G^>(w), and are otherwise the same
int fft_solver::pp_to_freq(GF &outG, const GF &inG, int set_dw)
{
	#ifndef STAND_ALONE
	CNTR_ASSERT_EQ(NESS_ASSERT_0, inG.size1_ , outG.size1_, __PRETTY_FUNCTION__);
	CNTR_ASSERT_EQ(NESS_ASSERT_0, inG.size2_ , outG.size2_, __PRETTY_FUNCTION__);
	CNTR_ASSERT_EQ(NESS_ASSERT_0, outG.ngrid_ , this -> nfreq_, __PRETTY_FUNCTION__);
	CNTR_ASSERT_EQ(NESS_ASSERT_0, inG.ngrid_ , this -> Nft_ / 2 + 1, __PRETTY_FUNCTION__);
	CNTR_ASSERT_EQ(NESS_ASSERT_0, inG.gf_type_ , time_gf, __PRETTY_FUNCTION__);
	CNTR_ASSERT_EQ(NESS_ASSERT_0, outG.gf_type_ , freq_gf, __PRETTY_FUNCTION__);
	#else
	assert(inG.size1_ == outG.size1_);
	assert(inG.size2_ == outG.size2_);
	assert(outG.ngrid_ == this -> nfreq_);
	assert(inG.ngrid_ == this -> Nft_ / 2 + 1);
	assert(inG.gf_type_ == time_gf);
	assert(outG.gf_type_ == freq_gf);
	#endif

	for(int idx1 = 0; idx1 < inG.size1_; idx1++)
	{
		for(int idx2 = 0; idx2 < inG.size2_; idx2++)
		{
			//compute gret
			//std::memset(in, 0, sizeof(fftw_complex) * Nft_);
			for(int w = 0; w < Nft_ / 2; w++)
			{
				in[w][0] = (*inG.p_ret(w, idx1, idx2)).real();
				in[w][1] = (*inG.p_ret(w, idx1, idx2)).imag();
				//in[Nft_ - w - 1][0] = -(*inG.p_ret(w + 1, idx2, idx1)).real();
				//in[Nft_ - w - 1][1] = (*inG.p_ret(w + 1, idx2, idx1)).imag();
			}
			for(int w = Nft_ / 2; w < Nft_; w++) {in[w][0] = 0.0; in[w][1] = 0.0;}

			fftw_execute(p_back);
			for(int w = 0; w < nfreq_ / 2; w++)
			{
				(*outG.p_ret(w, idx1, idx2)) = out[w][0] + out[w][1] * II;
				(*outG.p_ret(nfreq_ - w - 1, idx1, idx2)) = out[Nft_ - w - 1][0] + out[Nft_ - w - 1][1] * II;
			}

			//compute glss
			for(int w = 0; w < Nft_ / 2; w++)
			{
				in[w][0] = (*inG.p_les(w, idx1, idx2)).real();
				in[w][1] = (*inG.p_les(w, idx1, idx2)).imag();
				in[Nft_ - w - 1][0] = -(*inG.p_les(w + 1, idx2, idx1)).real();
				in[Nft_ - w - 1][1] = (*inG.p_les(w + 1, idx2, idx1)).imag();
			}
			in[Nft_ / 2][0] = (*inG.p_les(Nft_ / 2, idx2, idx1)).real();
			in[Nft_ / 2][1] = (*inG.p_les(Nft_ / 2, idx2, idx1)).imag();

			//in[Nft_ / 2][0] = in[Nft_ / 2 + 1][0];
			//in[Nft_ / 2][1] = in[Nft_ / 2 + 1][1];
			fftw_execute(p_back);
			for(int w = 0; w < nfreq_ / 2; w++)
			{
				(*outG.p_les(w, idx1, idx2)) = out[w][0] + out[w][1] * II;
				(*outG.p_les(nfreq_ - w - 1, idx1, idx2)) = out[Nft_ - w - 1][0] + out[Nft_ - w - 1][1] * II;
			}


		}
	}
	outG.smul(inG.dgrid_);
	if(set_dw) outG.reset_grid(2.0 * M_PI / (inG.dgrid_ * (Nft_ - 1)));
}


int fft_solver::pp_to_time(GF &outG, const GF &inG, int set_dt)
{
	#ifndef STAND_ALONE
	CNTR_ASSERT_EQ(NESS_ASSERT_0, inG.size1_ , outG.size1_, __PRETTY_FUNCTION__);
	CNTR_ASSERT_EQ(NESS_ASSERT_0, inG.size2_ , outG.size2_, __PRETTY_FUNCTION__);
	CNTR_ASSERT_EQ(NESS_ASSERT_0, inG.ngrid_ , this -> nfreq_, __PRETTY_FUNCTION__);
	CNTR_ASSERT_EQ(NESS_ASSERT_0, outG.ngrid_ , this -> Nft_ / 2 + 1, __PRETTY_FUNCTION__);
	CNTR_ASSERT_EQ(NESS_ASSERT_0, outG.gf_type_ , time_gf, __PRETTY_FUNCTION__);
	CNTR_ASSERT_EQ(NESS_ASSERT_0, inG.gf_type_ , freq_gf, __PRETTY_FUNCTION__);
	#else
	assert(inG.size1_ == outG.size1_);
	assert(inG.size2_ == outG.size2_);
	assert(inG.ngrid_ == this -> nfreq_);
	assert(outG.ngrid_ == this -> Nft_ / 2 + 1);
	assert(outG.gf_type_ == time_gf);
	assert(inG.gf_type_ == freq_gf);
	#endif

	
	for(int idx1 = 0; idx1 < inG.size1_; idx1++)
	{
		for(int idx2 = 0; idx2 < inG.size2_; idx2++)
		{
			//compute gret
			//for(int w = 0; w < Nft_; w++) {in[w][0] = 0.0; in[w][1] = 0.0;}
			for(int w = 0; w < nfreq_ / 2; w++)
			{
				in[w][0] = ((*inG.p_ret(w, idx1, idx2)).real() - (*inG.p_ret(w, idx2, idx1)).real());
				in[w][1] = ((*inG.p_ret(w, idx1, idx2)).imag() + (*inG.p_ret(w, idx2, idx1)).imag());
				in[Nft_ - w - 1][0] = ((*inG.p_ret(nfreq_ - w - 1, idx1, idx2)).real() - (*inG.p_ret(nfreq_ - w - 1, idx2, idx1)).real());
				in[Nft_ - w - 1][1] = ((*inG.p_ret(nfreq_ - w - 1, idx1, idx2)).imag() + (*inG.p_ret(nfreq_ - w - 1, idx2, idx1)).imag());

			}
			fftw_execute(p_for);
			for(int w = 0; w < Nft_ / 2; w++)
			{
				(*outG.p_ret(w, idx1, idx2)) = out[w][0] + II * out[w][1];
			}
			
			//for(int w = 0; w < Nft_; w++) {in[w][0] = 0.0; in[w][1] = 0.0;}

			for(int w = 0; w < nfreq_ / 2; w++)
			{
				in[w][0] = (*inG.p_les(w, idx1, idx2)).real();
				in[w][1] = (*inG.p_les(w, idx1, idx2)).imag();
				in[Nft_ - w - 1][0] = (*inG.p_les(nfreq_ - w - 1, idx1, idx2)).real();
				in[Nft_ - w - 1][1] = (*inG.p_les(nfreq_ - w - 1, idx1, idx2)).imag();
			}
			fftw_execute(p_for);
			for(int w = 0; w < Nft_ / 2 + 1; w++)
			{
				(*outG.p_les(w, idx1, idx2)) = out[w][0] + out[w][1] * II;
			}
			(*outG.p_ret(0, idx1, idx2)) *= 0.5;
		}
	}
	outG.smul(inG.dgrid_ / (2.0 * M_PI));
	if(set_dt) outG.reset_grid(2.0 * M_PI / (inG.dgrid_ * (Nft_ - 1)));
}
#endif


int fft_solver::to_freq(GF &outG, const GF &inG, int set_dw)
{
	#ifndef STAND_ALONE
	CNTR_ASSERT_EQ(NESS_ASSERT_0, inG.size1_ , outG.size1_, __PRETTY_FUNCTION__);
	CNTR_ASSERT_EQ(NESS_ASSERT_0, inG.size2_ , outG.size2_, __PRETTY_FUNCTION__);
	CNTR_ASSERT_EQ(NESS_ASSERT_0, outG.ngrid_ , this -> nfreq_, __PRETTY_FUNCTION__);
	CNTR_ASSERT_EQ(NESS_ASSERT_0, inG.ngrid_ , this -> Nft_ / 2 + 1, __PRETTY_FUNCTION__);
	CNTR_ASSERT_EQ(NESS_ASSERT_0, inG.gf_type_ , time_gf, __PRETTY_FUNCTION__);
	CNTR_ASSERT_EQ(NESS_ASSERT_0, outG.gf_type_ , freq_gf, __PRETTY_FUNCTION__);
	#else
	assert(inG.size1_ == outG.size1_);
	assert(inG.size2_ == outG.size2_);
	assert(outG.ngrid_ == this -> nfreq_);
	assert(inG.ngrid_ == this -> Nft_ / 2 + 1);
	assert(inG.gf_type_ == time_gf);
	assert(outG.gf_type_ == freq_gf);
	#endif

/*
	if (padded_result)
	{
		assert(outG.ngrid_ == Nft);
		assert(inG.ngrid_ == this -> Nft / 2);
	}
	else
	{
		assert(outG.ngrid_ == inG.ngrid_ * 2);
		assert(inG.ngrid_ == this -> ngrid_ / 2);
	}
*/	
	for(int idx1 = 0; idx1 < inG.size1_; idx1++)
	{
		for(int idx2 = 0; idx2 < inG.size2_; idx2++)
		{
			//compute gret
			//std::memset(in, 0, sizeof(fftw_complex) * Nft_);
			for(int w = 0; w < Nft_ / 2; w++)
			{
				in[w][0] = (*inG.p_ret(w, idx1, idx2)).real();
				in[w][1] = (*inG.p_ret(w, idx1, idx2)).imag();
				//in[Nft_ - w - 1][0] = 0;
				//in[Nft_ - w - 1][1] = 0;
			}
			for(int w = Nft_ / 2; w < Nft_; w++) {in[w][0] = 0.0; in[w][1] = 0.0;}
			fftw_execute(p_back);
			for(int w = 0; w < nfreq_ / 2; w++)
			{
				(*outG.p_ret(w, idx1, idx2)) = out[w][0] + out[w][1] * II;
				(*outG.p_ret(nfreq_ - w - 1, idx1, idx2)) = out[Nft_ - w - 1][0] + out[Nft_ - w - 1][1] * II;
			}

			//compute glss
			//std::memset(in, 0, sizeof(fftw_complex) * Nft_);
			for(int w = 0; w < Nft_ / 2; w++)
			{
				in[w][0] = (*inG.p_les(w, idx1, idx2)).real();
				in[w][1] = (*inG.p_les(w, idx1, idx2)).imag();
				//in[Nft_ - w - 1][0] = -std::conj(*inG.p_les(w + 1, idx2, idx1)).real();
				//in[Nft_ - w - 1][1] = -std::conj(*inG.p_les(w + 1, idx2, idx1)).imag();
				in[Nft_ - w - 1][0] = -(*inG.p_les(w + 1, idx2, idx1)).real();
				in[Nft_ - w - 1][1] = (*inG.p_les(w + 1, idx2, idx1)).imag();
			}
			in[Nft_ / 2][0] = (*inG.p_les(Nft_ / 2, idx2, idx1)).real();
			in[Nft_ / 2][1] = (*inG.p_les(Nft_ / 2, idx2, idx1)).imag();

			//in[Nft_ / 2][0] = in[Nft_ / 2 + 1][0];
			//in[Nft_ / 2][1] = in[Nft_ / 2 + 1][1];
			//in[Nft_ / 2][0] = (*inG.p_les(Nft_ / 2, idx1, idx2)).real();
			//in[Nft_ / 2][1] = (*inG.p_les(Nft_ / 2, idx1, idx2)).imag();
			fftw_execute(p_back);
			for(int w = 0; w < nfreq_ / 2; w++)
			{
				(*outG.p_les(w, idx1, idx2)) = out[w][0] + out[w][1] * II;
				(*outG.p_les(nfreq_ - w - 1, idx1, idx2)) = out[Nft_ - w - 1][0] + out[Nft_ - w - 1][1] * II;
			}


		}
	}
	//outG.smul(1.0 / (2.0 * M_PI * sqrt(Nft_)) * inG.dgrid_);
	outG.smul(inG.dgrid_);
	if(set_dw) outG.reset_grid(2.0 * M_PI / (inG.dgrid_ * (Nft_ - 1)));
	return 0;
}


int fft_solver::to_time(GF &outG, const GF &inG, int set_dt)
{

	#ifndef STAND_ALONE
	CNTR_ASSERT_EQ(NESS_ASSERT_0, inG.size1_ , outG.size1_, __PRETTY_FUNCTION__);
	CNTR_ASSERT_EQ(NESS_ASSERT_0, inG.size2_ , outG.size2_, __PRETTY_FUNCTION__);
	CNTR_ASSERT_EQ(NESS_ASSERT_0, inG.ngrid_ , this -> nfreq_, __PRETTY_FUNCTION__);
	CNTR_ASSERT_EQ(NESS_ASSERT_0, outG.ngrid_ , this -> Nft_ / 2 + 1, __PRETTY_FUNCTION__);
	CNTR_ASSERT_EQ(NESS_ASSERT_0, outG.gf_type_ , time_gf, __PRETTY_FUNCTION__);
	CNTR_ASSERT_EQ(NESS_ASSERT_0, inG.gf_type_ , freq_gf, __PRETTY_FUNCTION__);
	#else
	assert(inG.size1_ == outG.size1_);
	assert(inG.size2_ == outG.size2_);
	assert(inG.ngrid_ == this -> nfreq_);
	assert(outG.ngrid_ == this -> Nft_ / 2 + 1);
	assert(outG.gf_type_ == time_gf);
	assert(inG.gf_type_ == freq_gf);
	#endif

	
	for(int idx1 = 0; idx1 < inG.size1_; idx1++)
	{
		for(int idx2 = 0; idx2 < inG.size2_; idx2++)
		{
			//compute gret
			//std::memset(in, 0, sizeof(fftw_complex) * Nft_);
			for(int w = 0; w < Nft_; w++) {in[w][0] = 0.0; in[w][1] = 0.0;}
			for(int w = 0; w < nfreq_ / 2; w++)
			{
				in[w][0] = ((*inG.p_ret(w, idx1, idx2)).real() - (*inG.p_ret(w, idx2, idx1)).real()) + (*inG.p_les(w, idx1, idx2)).real();
				in[w][1] = ((*inG.p_ret(w, idx1, idx2)).imag() + (*inG.p_ret(w, idx2, idx1)).imag()) + (*inG.p_les(w, idx1, idx2)).imag();
				in[Nft_ - w - 1][0] = ((*inG.p_ret(nfreq_ - w - 1, idx1, idx2)).real() - (*inG.p_ret(nfreq_ - w - 1, idx2, idx1)).real()) + (*inG.p_les(nfreq_ - w - 1, idx1, idx2)).real();
				in[Nft_ - w - 1][1] = ((*inG.p_ret(nfreq_ - w - 1, idx1, idx2)).imag() + (*inG.p_ret(nfreq_ - w - 1, idx2, idx1)).imag()) + (*inG.p_les(nfreq_ - w - 1, idx1, idx2)).imag();
			/*
				in[w][0] = (*inG.p_ret(w, idx1, idx2)).real();
				in[w][1] = (*inG.p_ret(w, idx1, idx2)).imag();
				in[Nft_ - w - 1][0] = (*inG.p_ret(nfreq_ - w - 1, idx1, idx2)).real();
				in[Nft_ - w - 1][1] = (*inG.p_ret(nfreq_ - w - 1, idx1, idx2)).imag();
			*/

			}
			fftw_execute(p_for);
			//we have to store out[Nft_/2]
			for(int w = 0; w < Nft_ / 2; w++)
			{
				(*outG.p_ret(w, idx1, idx2)) = out[w][0] + II * out[w][1];
			}
			//for(int w = 1; w < nfreq_ - 1; w++)
				//std::cout << idx1 << "\t" << idx2 << "\t" << w << '\t' << out[w][0] + out[w][1] * II << '\t' << out[Nft_ - w][0] + II * out[Nft_ - w][1] << std::endl;

			//compute glss
			//std::memset(in, 0, sizeof(fftw_complex) * Nft_);
			for(int w = 0; w < Nft_; w++) {in[w][0] = 0.0; in[w][1] = 0.0;}
			for(int w = 0; w < nfreq_ / 2; w++)
			{
				in[w][0] = (*inG.p_les(w, idx1, idx2)).real();
				in[w][1] = (*inG.p_les(w, idx1, idx2)).imag();
				in[Nft_ - w - 1][0] = (*inG.p_les(nfreq_ - w - 1, idx1, idx2)).real();
				in[Nft_ - w - 1][1] = (*inG.p_les(nfreq_ - w - 1, idx1, idx2)).imag();
			}
			fftw_execute(p_for);
			for(int w = 0; w < Nft_ / 2 + 1; w++)
			{
				(*outG.p_les(w, idx1, idx2)) = out[w][0] + out[w][1] * II;
				(*outG.p_ret(w, idx1, idx2)) -= (*outG.p_les(w, idx1, idx2));
			}
			(*outG.p_ret(Nft_ / 2, idx1, idx2)) = 0.0;
			#if FACTOR_TWO_MAGIC==1
                (*outG.p_ret(0, idx1, idx2)) *= 0.5;
            #endif
		}
	}
	//outG.smul(1.0 / (2.0 * M_PI * sqrt(Nft_)) * inG.dgrid_);
	outG.smul(inG.dgrid_ / (2.0 * M_PI));
	/// CHECK THIS ????
    //if(set_dt) outG.reset_grid(2.0 * M_PI / (inG.dgrid_ * (Nft_ - 1)));
    if(set_dt) outG.reset_grid(2.0 * M_PI / (inG.dgrid_ * (Nft_)));
 	return 0;   
}

int fft_solver::to_time_corfac(GF &outG, const GF &inG){
    fft_solver::to_time(outG,inG,1);
    compute_corfac();
    for(int i=0;i<outG.ngrid_;i++) {
        //std::cout << "Nft= " << Nft_ << " i " << i << " c: " << corfac[i] << std::endl;
        outG.Retarded[i]*=corfac[i];
        outG.Lesser[i]*=corfac[i];
    }
	return 0;
}



// CORRECTION FACTOR WAS JUST A TRY ... IT DIOES NOT IMPROVE THE RESULTS
double get_fft_corfac(double theta)
{
	double corfac;
    double t,t2,t4,t6;
	double cth,ctth,spth2,sth,sth4i,stth,th,th2,th4,tmth2,tth4i;
	th=theta;
	if (fabs(th) < 5.0e-2) {
		t=th;
		t2=t*t;
		t4=t2*t2;
		t6=t4*t2;
		corfac=1.0-(11.0/720.0)*t4+(23.0/15120.0)*t6;
	} else {
		cth=cos(th);
		sth=sin(th);
		ctth=cth*cth-sth*sth;
		stth=2.0e0*sth*cth;
		th2=th*th;
		th4=th2*th2;
		tmth2=3.0e0-th2;
		spth2=6.0e0+th2;
		sth4i=1.0/(6.0e0*th4);
		tth4i=2.0e0*sth4i;
		corfac=tth4i*spth2*(3.0e0-4.0e0*cth+ctth);
	}
    return corfac;
}
void fft_solver::compute_corfac(void){
    if(corfac_exists==false){
        // this is done only once!
        corfac = new double [Nft_];
        for(int i=0;i<Nft_;i++){
            double theta=(i*2.0*M_PI)/(Nft_ - 1);
            corfac[i]=get_fft_corfac(theta);
            //std::cout << "Nft= " << Nft_ << " i " << i << " theta : " << theta << " c: " << corfac[i] << std::endl;
        }
        corfac_exists=true;
        //exit(0);
    }
}






};
#endif
