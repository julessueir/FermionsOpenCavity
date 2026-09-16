#ifndef FFT_WEIGHT_IMPL
#define FFT_WEIGHT_IMPL

#ifndef STAND_ALONE
	#include "cntr.hpp"
#endif
//#include "ness_GF_decl.hpp"
#include "ness_fft_decl.hpp"


namespace ness {

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
	if(set_dt) outG.reset_grid(2.0 * M_PI / (inG.dgrid_ * (Nft_ - 1)));
}
};
#endif
