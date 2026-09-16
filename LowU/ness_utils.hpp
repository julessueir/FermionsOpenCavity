//Jiajun Li, Anne Matthies 2018
#ifndef UTILS
#define UTILS
//#include "ness_GF_decl.hpp"
#include "ness_global_settings.hpp"
#include <fstream>
namespace ness
{
void force_FD(GF &gf, double temp)
{
	for(int w = 0; w < gf.ngrid_; w++)
	{
		double fd = -tanh(gf.grid_[w] / temp)/2.0 + 0.5;
		gf.Lesser[w] = -(gf.Retarded[w] - gf.Retarded[w].adjoint()) * fd;
	}

}

void force_imag(GF &gf)
{
	if(gf.gf_type_ == time_gf)
		return;
	for(int w = 0; w < gf.ngrid_; w++)
	{
		gf.Lesser[w] = 0.5 * (gf.Lesser[w] - gf.Lesser[w].adjoint());
	}

}

int set_bwd_from_fwd(GF & gf_bwd, GF & gf_fwd)
{
	#ifndef STAND_ALONE
	CNTR_ASSERT_EQ(NESS_ASSERT_0, gf_fwd.ngrid_ , gf_bwd.ngrid_, __PRETTY_FUNCTION__);
	CNTR_ASSERT_EQ(NESS_ASSERT_0, gf_fwd.size1_, gf_bwd.size1_, __PRETTY_FUNCTION__);
	CNTR_ASSERT_EQ(NESS_ASSERT_0, gf_fwd.size2_, gf_bwd.size2_, __PRETTY_FUNCTION__);
	#else
	assert(gf_fwd.ngrid_ == gf_bwd.ngrid_);
	assert(gf_fwd.size1_ == gf_bwd.size1_);
	assert(gf_fwd.size2_ == gf_bwd.size2_);
	#endif

	for(int w = 0; w < gf_fwd.ngrid_; w++)
	{
		//gf_bwd.Lesser[w] = -gf_fwd.Lesser[w].adjoint();
		//gf_bwd.Retarded[w] = -gf_fwd.Greater(w).adjoint() - gf_bwd.Lesser[w]; // This is to store ggtr (gret is actually zero for t < 0)
		gf_bwd.Lesser[w] = -gf_fwd.Greater(w).adjoint();
		gf_bwd.Retarded[w] = -gf_fwd.Lesser[w].adjoint() - gf_bwd.Lesser[w]; // This is to store ggtr (gret is actually zero for t < 0)
	}
	gf_bwd.Retarded[0] *= 0.5;
	return 0;
}

int dyson(GF &invg0, GF &self, GF &g)
{
	assert(self.data_size_ == invg0.data_size_);
	cplx temp[self.el_size_];
	cplx tempc[self.el_size_];
	//gret
	for (int w = 0; w < self.ngrid_; w++)
	{
		#ifndef STAND_ALONE	
		int ind0 = w * g.el_size_;
		for(int ind = 0; ind < self.el_size_; ind++)
			tempc[ind] = invg0.g_ret[ind0 + ind] - self.g_ret[ind0 + ind];
		cntr::element_inverse<double, LARGESIZE>(g.size1_, g.g_ret + w * g.el_size_, tempc);
		#else
		g.Retarded[w] = invg0.Retarded[w] - self.Retarded[w];
		#endif
	}
	//glss
	for (int w = 0; w < g.ngrid_; w++)
	{
		#ifndef STAND_ALONE
		int ind = w * g.el_size_;
		cntr::element_mult<double, LARGESIZE> (g.size1_, temp, g.g_ret + ind, self.g_les + ind);
		cntr::element_conj<double, LARGESIZE> (g.size1_, tempc, g.g_ret + ind);
		cntr::element_mult<double, LARGESIZE> (g.size1_, g.g_les + ind, temp, tempc);
		#else
		g.Lesser[w] = g.Retarded[w] * self.Lesser[w] * g.Retarded[w].adjoint();
		#endif
	}
	return 0;
}


int dyson_from_inv(GF &g, GF &self)
{
	assert(self.data_size_ == g.data_size_);
	assert(self.size1_ == g.size1_);
	assert(self.size2_ == g.size2_);

	cplx temp[self.el_size_];
	cplx tempc[self.el_size_];
	for (int w = 0; w < self.ngrid_; w++)
	{
		//g.Retarded[w] = (g.Retarded[w] - self.Retarded[w]).inverse().eval();
		//g.Lesser[w] = g.Retarded[w] * self.Lesser[w] * g.Retarded[w].adjoint();
		//gret
		#ifndef STAND_ALONE
		int ind0 = w * g.el_size_;
		for(int ind = 0; ind < self.el_size_; ind++)
			tempc[ind] = g.g_ret[ind0 + ind] - self.g_ret[ind0 + ind];
		cntr::element_inverse<double, LARGESIZE>(self.size1_, g.g_ret + w * g.el_size_, tempc);
		//glss
		cntr::element_mult<double, LARGESIZE> (g.size1_, temp, g.g_ret + ind0, self.g_les + ind0);
		cntr::element_conj<double, LARGESIZE> (g.size1_, tempc, g.g_ret + ind0);
		cntr::element_mult<double, LARGESIZE> (g.size1_, g.g_les + ind0, temp, tempc);
		#else
			g.Retarded[w] = (g.Retarded[w] - self.Retarded[w]).inverse();
			g.Lesser[w] = g.Retarded[w] * self.Lesser[w] * g.Retarded[w].adjoint();
		#endif

	}
	return 0;
}

int invert_invG(GF &g, GF &self)
{
	assert(self.data_size_ == g.data_size_);
	assert(self.size1_ == g.size1_);
	assert(self.size2_ == g.size2_);

	cplx temp[self.el_size_];
	cplx tempc[self.el_size_];
	for (int w = 0; w < self.ngrid_; w++)
	{
		//gret
		#ifndef STAND_ALONE
		int ind0 = w * g.el_size_;
		for(int ind = 0; ind < self.el_size_; ind++)
			tempc[ind] = g.g_ret[ind0 + ind];
		cntr::element_inverse<double, LARGESIZE>(self.size1_, g.g_ret + w * g.el_size_, tempc);
		//glss
		cntr::element_mult<double, LARGESIZE> (g.size1_, temp, g.g_ret + ind0, self.g_les + ind0);
		cntr::element_conj<double, LARGESIZE> (g.size1_, tempc, g.g_ret + ind0);
		cntr::element_mult<double, LARGESIZE> (g.size1_, g.g_les + ind0, temp, tempc);
		#else
			g.Retarded[w] = (g.Retarded[w] - self.Retarded[w]).inverse();
			g.Lesser[w] = g.Retarded[w] * self.Lesser[w] * g.Retarded[w].adjoint();
		#endif

	}
	return 0;
}

double GF2norm(GF &g1, GF &g2)
{
	double norm = 0.0;
	assert(g1.data_size_ == g2.data_size_);

	for (int w = 0; w < g1.data_size_ ; w++)
	{
		norm += std::norm(g1.g_ret[w] - g2.g_ret[w]);
		norm += std::norm(g1.g_les[w] - g2.g_les[w]);
	}
	norm = sqrt(norm) / (double) g1.data_size_;

	return norm;
}

void outputGF(GF &G, std::string filename){
	std::ofstream outputFile;
	outputFile.open(filename.c_str());
	outputFile << "# frequency" << "\t" << "Spectrum" << "\t"<< "Occupation" << "\t"<< "Retarded Real" << "\t"<< "Retarded imag"<< "\t"<< "Lesser imag"<< "\t"<< "Lesser real"<< std::endl;
	long N=G.ngrid_;

	for (long w = 0; w < N; w++)
	{
		int i = w * G.el_size_;
		outputFile <<std::setprecision(15) << G.grid_[w] << "\t" << G.g_ret[i].real() << "\t"<< G.g_ret[i].imag() << "\t"<< G.g_les[i].imag() << "\t"<< G.g_les[i].real() << "\t"
			   << G.g_ret[i+1].real() << "\t"<< G.g_ret[i+1].imag() << "\t"<< G.g_les[i+1].imag() << "\t"<< G.g_les[i+1].real() << "\t" 
			   << G.g_ret[i+2].real() << "\t"<< G.g_ret[i+2].imag() << "\t"<< G.g_les[i+2].imag() << "\t"<< G.g_les[i+2].real() << "\t"
			   << G.g_ret[i+3].real() << "\t"<< G.g_ret[i+3].imag() << "\t"<< G.g_les[i+3].imag() << "\t"<< G.g_les[i+3].real() << std::endl;
	}
	
	outputFile.close();
}


void outputGF_new(GF &G, std::string filename){
	std::ofstream outputFile;
	outputFile.open(filename.c_str());
	outputFile << "# frequency" << "\t" <<  "Retarded Real" << "\t"<< "Retarded imag"<< "\t"<< "Lesser real"<< "\t"<< "Lesser imag"<< std::endl;
	long N=G.ngrid_;
    outputFile <<std::setprecision(20); 
	for (long w = 0; w < N; w++)
	{
        outputFile << G.grid_[w] << " ";
        for(int i=0;i<G.el_size_;i++){
            long idx=w * G.el_size_+i;
            if (i==0 or i==G.el_size_-1) outputFile << G.g_ret[idx].real() << " "<< G.g_ret[idx].imag() << " "; //I want to print just the 00 and 11 components
        }
        for(int i=0;i<G.el_size_;i++){
            long idx=w * G.el_size_+i;
             if (i==0 or i==G.el_size_-1) outputFile << G.g_les[idx].real() << " "<< G.g_les[idx].imag() << " "; //I want to print just the 00 and 11 components

        }
        outputFile << "\n";
	}
	outputFile.close();
}

void outputGF_new_F(GF &G, GF &F, std::string filename){
	std::ofstream outputFile;
	outputFile.open(filename.c_str());
	outputFile << "# frequency" << "\t" <<  "Retarded Real" << "\t"<< "Retarded imag"<< "\t"<< "Lesser real"<< "\t"<< "Lesser imag"<< std::endl;
	long N=G.ngrid_;
    outputFile <<std::setprecision(20); 
	for (long w = 0; w < N; w++)
	{
        outputFile << G.grid_[w] << " ";
        for(int i=0;i<G.el_size_;i++){
            long idx=w * G.el_size_+i;
            if (i==0 or i==G.el_size_-1) outputFile << G.g_ret[idx].real() << " "<< G.g_ret[idx].imag() << " "; //I want to print just the 00 and 11 components
        }
        for(int i=0;i<G.el_size_;i++){
            long idx=w * G.el_size_+i;
             if (i==0 or i==G.el_size_-1) outputFile << G.g_les[idx].real() << " "<< G.g_les[idx].imag() << " "; //I want to print just the 00 and 11 components

        }

		for(int i=0;i<G.el_size_;i++){
            long idx=w * G.el_size_+i;
             if (i==0 or i==G.el_size_-1) outputFile << F.g_les[idx].real() << " "<< F.g_les[idx].imag() << " "; //I want to print just the 00 and 11 components
        }


        outputFile << "\n";
	}
	outputFile.close();
}


void outputGF_reduced(GF &G, std::string filename) {
    std::ofstream outputFile;
    outputFile.open(filename.c_str());
    outputFile << "# frequency" << "\t" << "Retarded Real" << "\t" << "Retarded Imag" << "\t" 
               << "Lesser Real" << "\t" << "Lesser Imag" << std::endl;

    long N = G.ngrid_;
    long segment_size = N / 20; // Number of points to keep from start and end

    outputFile << std::setprecision(20);

    // First segment: from start up to the first N/20 points
    for (long w = 0; w < segment_size; w += 10) {
        if (G.grid_[w] > 5.0) break; // Stop if we exceed the required range
        outputFile << G.grid_[w] << " ";
        for (int i = 0; i < G.el_size_; i++) {
            long idx = w * G.el_size_ + i;
            if (i == 0 or i == G.el_size_ - 1) outputFile << G.g_ret[idx].real() << " " << G.g_ret[idx].imag() << " ";
        }
        for (int i = 0; i < G.el_size_; i++) {
            long idx = w * G.el_size_ + i;
            if (i == 0 or i == G.el_size_ - 1) outputFile << G.g_les[idx].real() << " " << G.g_les[idx].imag() << " ";
        }
        outputFile << "\n";
    }

    // Last segment: from N - N/20 to N
    for (long w = N - segment_size; w < N; w += 10) {
        if (G.grid_[w] < -5.0) continue; // Skip values outside required range
        outputFile << G.grid_[w] << " ";
        for (int i = 0; i < G.el_size_; i++) {
            long idx = w * G.el_size_ + i;
            if (i == 0 or i == G.el_size_ - 1) outputFile << G.g_ret[idx].real() << " " << G.g_ret[idx].imag() << " ";
        }
        for (int i = 0; i < G.el_size_; i++) {
            long idx = w * G.el_size_ + i;
            if (i == 0 or i == G.el_size_ - 1) outputFile << G.g_les[idx].real() << " " << G.g_les[idx].imag() << " ";
        }
        outputFile << "\n";
    }

    outputFile.close();
}



/*
void printGF(GF &G, std::string filename){
	std::ofstream outputFile;
	outputFile.open(filename.c_str());
	outputFile << "# frequency" << "\t" <<  "Retarded Real" << "\t"<< "Retarded imag"<< "\t"<< "Lesser real"<< "\t"<< "Lesser imag"<< std::endl;
	long N=G.ngrid_;
    outputFile <<std::setprecision(15);
	for (long w = 0; w < N; w++)
	{
        outputFile << G.grid_[w] << " ";
        for(int i=0;i<G.el_size_;i++){
            long idx=w * G.el_size_+i;
            outputFile << G.g_ret[idx].real() << " "<< G.g_ret[idx].imag() << " "; //I want to print just the 00 and 11 components
        }
        for(int i=0;i<G.el_size_;i++){
            long idx=w * G.el_size_+i;
            outputFile << G.g_les[idx].real() << " "<< G.g_les[idx].imag() << " "; //I want to print just the 00 and 11 components
        }
        outputFile << "\n";
	}
	outputFile.close();
}
*/

int healthCheck(GF &G){
	//for( auto idx : ppsc::range(0, G.data_size_))
	for(int idx = 0; idx < G.data_size_; idx++)
	{
		if(G.g_ret[idx] != G.g_ret[idx] || G.g_les[idx] != G.g_les[idx])
			return idx;


	}
	return -1;
}
};//end of namespace
#endif
