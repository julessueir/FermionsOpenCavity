#ifndef GF_IMPL
#define GF_IMPL

//Jiajun Li, Anne Matthies 2018

#include "ness_GF_decl.hpp"


namespace ness
{

int GF::clear()
{
	std::memset(g_ret, 0, sizeof(cplx) * data_size_);
	std::memset(g_les, 0, sizeof(cplx) * data_size_);
	return 0;
}

//template<class Matrix>
int GF::left_multiply(const cdmatrix &g, double weight)
{
	cplx xtemp[el_size_];
	cplx mat[el_size_];
	for (int idx1 = 0; idx1 < size1_; idx1++)
	{
		for (int idx2 = 0; idx2 < size2_; idx2++)
		{
			mat[idx1 * size2_ + idx2] = g(idx1, idx2);
		}
	}

	cplx *les_ptr, *ret_ptr;
	for (int w = 0; w < ngrid_; w++)
	{
		#ifndef STAND_ALONE
		ret_ptr = g_ret + w * el_size_;
		cntr::element_mult<double, LARGESIZE>(size1_, xtemp, mat, ret_ptr);
		cntr::element_smul<double, LARGESIZE>(size1_, xtemp, weight);
		cntr::element_set<double, LARGESIZE>(size1_, ret_ptr, xtemp);

		les_ptr = g_les + w * el_size_;
		cntr::element_mult<double, LARGESIZE>(size1_, xtemp, mat, les_ptr);
		cntr::element_smul<double, LARGESIZE>(size1_, xtemp, weight);
		cntr::element_set<double, LARGESIZE>(size1_, les_ptr, xtemp);
		#else
		Retarded[w] = (g * Retarded[w]).eval();
		Lesser[w] = (g * Lesser[w]).eval();
		#endif
	}
	return 0;
}

//template<class Matrix>
int GF::right_multiply(const cdmatrix &g, double weight)
{
	cplx xtemp[el_size_];
	cplx mat[el_size_];
	for (int idx1 = 0; idx1 < size1_; idx1++)
	{
		for (int idx2 = 0; idx2 < size2_; idx2++)
		{
			mat[idx1 * size2_ + idx2] = g(idx1, idx2);
		}
	}

	cplx *les_ptr, *ret_ptr;
	for (int w = 0; w < ngrid_; w++)
	{
		#ifndef STAND_ALONE
		ret_ptr = g_ret + w * el_size_;
		cntr::element_mult<double, LARGESIZE>(size1_, xtemp, ret_ptr, mat);
		cntr::element_smul<double, LARGESIZE>(size1_, xtemp, weight);
		cntr::element_set<double, LARGESIZE>(size1_, ret_ptr, xtemp);

		les_ptr = g_les + w * el_size_;
		cntr::element_mult<double, LARGESIZE>(size1_, xtemp, les_ptr, mat);
		cntr::element_smul<double, LARGESIZE>(size1_, xtemp, weight);
		cntr::element_set<double, LARGESIZE>(size1_, les_ptr, xtemp);
		#else
		Retarded[w] = (Retarded[w] * g).eval();
		Lesser[w] = (Lesser[w] * g).eval();
		#endif

	}
	return 0;
}

int GF::set_shiftGF(int shift)
{

	cdmatrix zerom = Eigen::MatrixXcd::Zero(size1_, size2_);
	int ashift = std::abs(shift);
	if (shift > 0)
	{

		for(int w = ngrid_ / 2 + 1; w < ngrid_ - shift; w++)
		{
			Retarded[w] = Retarded[w + shift];
			Lesser[w] = Lesser[w + shift];
		}
		for(int w = ngrid_ - shift; w < ngrid_; w++)
		{
			Retarded[w] = Retarded[w - ngrid_ + shift];
			Lesser[w] = Lesser[w - ngrid_ + shift];
		}

		for(int w = 0; w < ngrid_ / 2 + 1 - shift; w++)
		{
			Retarded[w] = Retarded[w + shift];
			Lesser[w] = Lesser[w + shift];
		}
		for(int w = ngrid_ / 2 - shift + 1; w < ngrid_ / 2 + 1; w++)
		{
			Retarded[w] = zerom;
			Lesser[w] = zerom;
		}



	}
	else if (shift < 0)
	{

		for(int w = ngrid_ / 2; w >= ashift; w--)
		{
			Retarded[w] = Retarded[w - ashift];
			Lesser[w] = Lesser[w - ashift];
		}
		for(int w = ashift - 1; w >= 0; w--)
		{
			Retarded[w] = Retarded[ngrid_ + w - ashift];
			Lesser[w] = Lesser[ngrid_ + w - ashift];
		}

		for(int w = ngrid_ - 1; w >= ngrid_ / 2 + 1 + ashift; w--)
		{
			Retarded[w] = Retarded[w - ashift];
			Lesser[w] = Lesser[w - ashift];
		}
		for(int w = ngrid_ / 2 + ashift ; w >= ngrid_ / 2 + 1; w--)
		{
			Retarded[w] = zerom;
			Lesser[w] = zerom;
		}


	}

	return 0;
}

int GF::left_multiply(GF &g, double weight)
{
	assert(size1_ == g.size1_);
	assert(size2_ == g.size2_);
	assert(ngrid_ == g.ngrid_);
	cplx xtemp[el_size_];
	cplx xtemp2[el_size_];
	cplx xtemp3[el_size_];

	cplx *les_ptr, *ret_ptr;
	for (int w = 0; w < ngrid_; w++)
	{
		#ifndef STAND_ALONE
		ret_ptr = g_ret + w * el_size_;
		cntr::element_mult<double, LARGESIZE>(size1_, xtemp, g.p_ret(w,0,0), ret_ptr);
		cntr::element_smul<double, LARGESIZE>(size1_, xtemp, weight);

		//Langreth rule
		les_ptr = g_les + w * el_size_;
		cntr::element_mult<double, LARGESIZE>(size1_, xtemp2, g.p_ret(w,0,0), les_ptr);
		cntr::element_set<double, LARGESIZE>(size1_, les_ptr, xtemp2);
		cntr::element_conj<double, LARGESIZE>(size1_, xtemp2, ret_ptr);
		cntr::element_mult<double, LARGESIZE>(size1_, xtemp3, g.p_les(w,0,0), xtemp2);

		cntr::element_incr<double, LARGESIZE>(size1_, les_ptr, xtemp3);
		cntr::element_smul<double, LARGESIZE>(size1_, les_ptr, weight);
		cntr::element_set<double, LARGESIZE>(size1_, ret_ptr, xtemp);
		#else
		Retarded[w] = g.Retarded[w] * Retarded[w];
		Lesser[w] = g.Retarded[w] * Lesser[w] + g.Lesser[w] * Retarded[w].adjoint();
		#endif
	}
	return 0;
}


int GF::right_multiply(GF &g, double weight)
{
	assert(size1_ == g.size1_);
	assert(size2_ == g.size2_);
	assert(ngrid_ == g.ngrid_);
	cplx xtemp[el_size_];
	cplx xtemp2[el_size_];
	cplx xtemp3[el_size_];

	cplx *les_ptr, *ret_ptr;
	for (int w = 0; w < ngrid_; w++)
	{
		#ifndef STAND_ALONE
		ret_ptr = g_ret + w * el_size_;
		cntr::element_mult<double, LARGESIZE>(size1_, xtemp, ret_ptr, g.p_ret(w,0,0));
		cntr::element_smul<double, LARGESIZE>(size1_, xtemp, weight);

		//Langreth rule
		les_ptr = g_les + w * el_size_;
		cntr::element_mult<double, LARGESIZE>(size1_, xtemp2, ret_ptr, g.p_les(w,0,0));
		cntr::element_set<double, LARGESIZE>(size1_, les_ptr, xtemp2);
		cntr::element_conj<double, LARGESIZE>(size1_, xtemp2, g.p_ret(w,0,0));
		cntr::element_mult<double, LARGESIZE>(size1_, xtemp3, les_ptr, xtemp2);

		cntr::element_incr<double, LARGESIZE>(size1_, les_ptr, xtemp3);
		cntr::element_smul<double, LARGESIZE>(size1_, les_ptr, weight);
		cntr::element_set<double, LARGESIZE>(size1_, ret_ptr, xtemp);
		#else
		Retarded[w] = Retarded[w] * g.Retarded[w];
		Lesser[w] = Retarded[w] * g.Lesser[w] + Lesser[w] * g.Retarded[w].adjoint();
		#endif

	}
	return 0;
}

GF::~GF(){
	delete[] g_ret;
	delete[] g_les;
	delete[] grid_;
	//delete[] Retarded.data();
	//delete[] Lesser.data();
	/*
	for(int w = 0; w < ngrid_; w++)
	{
		Retarded[w].~Map();
		Lesser[w].~Map();
	}
	*/
}

GF::GF(){
	dgrid_ = 0.0;
	ngrid_ = 0;
	size1_ = 0;
	size2_ = 0;
	el_size_ = 0;
	data_size_ = 0;
	g_ret = NULL;
	g_les = NULL;
	grid_ = NULL;
	gf_type_ = 0;
	//cdmatrix zero2cd=cdmatrix::Zero();
	//grid_spacing=0.0;
	//startFreq=0.0;
	//numOfFreq=0;
	//Retarded.assign(1,zero2cd);
	//Lesser.assign(1,zero2cd);
}

int GF::incr(GF &gf, cplx weight)
{
	assert(this -> gf_type_ == gf.gf_type_);
	assert(this -> el_size_ == gf.el_size_);
	assert(this -> data_size_ == gf.data_size_);
	for (int idx = 0; idx < data_size_; idx++)
	{
		g_ret[idx] += gf.g_ret[idx] * weight;
		g_les[idx] += gf.g_les[idx] * weight;
	}
	return 0;
}

int GF::smul(cplx weight)
{
	for (int idx = 0; idx < data_size_; idx++)
	{
		g_ret[idx] *= weight;
		g_les[idx] *= weight;
	}
	return 0;
}

GF::GF(const GF &g)
{
	size1_ = g.size1_;
	size2_ = g.size2_;
	el_size_ = g.el_size_;
	dgrid_ = g.dgrid_;
	ngrid_ = g.ngrid_;
	gf_type_ = g.gf_type_;
	data_size_ = g.data_size_;

	g_ret = new cplx[data_size_];
	g_les = new cplx[data_size_];
	grid_ = new double[ngrid_];
	std::memcpy(g_ret, g.g_ret, sizeof(cplx) * data_size_);
	std::memcpy(g_les, g.g_les, sizeof(cplx) * data_size_);
	std::memcpy(grid_, g.grid_, sizeof(double) * ngrid_);
	Retarded.clear();
	Lesser.clear();

	Retarded.reserve(ngrid_);
	Lesser.reserve(ngrid_);
	for(int w = 0; w < ngrid_; w++)
	{
		Retarded.emplace_back(p_ret(w,0,0), size1_, size2_);
		Lesser.emplace_back(p_les(w,0,0), size1_, size2_);
	}
}

GF& GF::operator=(const GF &g)
{
	size1_ = g.size1_;
	size2_ = g.size2_;
	el_size_ = g.el_size_;
	dgrid_ = g.dgrid_;
	gf_type_ = g.gf_type_;
	if (this -> ngrid_ != g.ngrid_)
	{
		if (ngrid_ != 0)
			delete[] grid_;
		ngrid_ = g.ngrid_;
		grid_ = new double[ngrid_];

	}
	Retarded.clear();
	Lesser.clear();
	Retarded.reserve(ngrid_);
	Lesser.reserve(ngrid_);

	if (this -> data_size_!= g.data_size_) 
	{
		data_size_ = g.data_size_;
		if (data_size_ != 0)
		{
			delete[] g_ret;
			delete[] g_les;
		}
		g_ret = new cplx[data_size_];
		g_les = new cplx[data_size_];
	}
	std::memcpy(g_ret, g.g_ret, sizeof(cplx) * data_size_);
	std::memcpy(g_les, g.g_les, sizeof(cplx) * data_size_);
	std::memcpy(grid_, g.grid_, sizeof(double) * ngrid_);
	for(int w = 0; w < ngrid_; w++)
	{
		//Retarded.emplace_back(Eigen::Map<Eigen::MatrixXcd, Eigen::RowMajor> (p_ret(w,0,0), size1_, size2_));
		//Lesser.emplace_back(Eigen::Map<Eigen::MatrixXcd, Eigen::RowMajor> (p_les(w,0,0), size1_, size2_));
		Retarded.emplace_back(p_ret(w,0,0), size1_, size2_);
		Lesser.emplace_back(p_les(w,0,0), size1_, size2_);
	}


	return *this;
}

GF::GF(double dgrid, int ngrid, int size1, int size2, int gf_type):
size1_(size1), size2_(size2),
dgrid_(dgrid), ngrid_(ngrid), gf_type_(gf_type)
{
	Retarded.reserve(ngrid_);
	Lesser.reserve(ngrid_);
	int halfn = ngrid / 2;
	el_size_ = size1_ * size2_;
	// the grid is constructed in a way that negative branch is from w = halfn to ngrid_ - 1 
	// (note ngrid_ is implicitly even, totally ngrid_/2 points) and w = 0 is 0;
	// positive branch is from w = 1 to w = ngrid_ - 1 
	// (totally ngrid_/2 - 1 points);
	// important when doing fourier transform;
	grid_ = new double[ngrid_];
	if(gf_type == freq_gf)
	{
		for (int w_ind = 0; w_ind < halfn; w_ind++)
		{
			grid_[w_ind] = w_ind * dgrid_;
			grid_[ngrid_ - w_ind - 1] =  -(w_ind + 1) * dgrid_;
		}
	}
	else if(gf_type == time_gf)
	{
		for (int w_ind = 0; w_ind < ngrid_; w_ind++)
		{
			grid_[w_ind] = w_ind * dgrid_;
		}
	}
	data_size_ = ngrid_ * el_size_;
	g_ret = new cplx[data_size_];
	g_les = new cplx[data_size_];
	std::memset(g_ret, 0, sizeof(cplx) * data_size_);
	std::memset(g_les, 0, sizeof(cplx) * data_size_);
	for(int w = 0; w < ngrid_; w++)
	{
		Retarded.emplace_back(Eigen::Map<Eigen::MatrixXcd, Eigen::RowMajor> (p_ret(w,0,0), size1_, size2_));
		Lesser.emplace_back(Eigen::Map<Eigen::MatrixXcd, Eigen::RowMajor> (p_les(w,0,0), size1_, size2_));
	}


}

GF::GF(double dgrid, int ngrid, int size1, int gf_type):
size1_(size1), size2_(size1), 
dgrid_(dgrid), ngrid_(ngrid), gf_type_(gf_type)
{
	Retarded.reserve(ngrid_);
	Lesser.reserve(ngrid_);
	int halfn = ngrid / 2;
	el_size_ = size1_ * size2_;
	//same way of constructing grid as above
	grid_ = new double[ngrid_];
	if(gf_type == freq_gf)
	{
		for (int w_ind = 0; w_ind < halfn; w_ind++)
		{
			grid_[w_ind] = w_ind * dgrid_;
			grid_[ngrid_ - w_ind - 1] =  -(w_ind + 1) * dgrid_;
		}
	}
	else if(gf_type == time_gf)
	{
		for (int w_ind = 0; w_ind < ngrid_; w_ind++)
		{
			grid_[w_ind] = w_ind * dgrid_;
		}
	}

	data_size_ = ngrid_ * el_size_;
	g_ret = new cplx[data_size_];
	g_les = new cplx[data_size_];
	std::memset(g_ret, 0, sizeof(cplx) * data_size_);
	std::memset(g_les, 0, sizeof(cplx) * data_size_);
	for(int w = 0; w < ngrid_; w++)
	{
		Retarded.emplace_back(Eigen::Map<Eigen::MatrixXcd, Eigen::RowMajor> (p_ret(w,0,0), size1_, size2_));
		Lesser.emplace_back(Eigen::Map<Eigen::MatrixXcd, Eigen::RowMajor> (p_les(w,0,0), size1_, size2_));

	}

}

int GF::self_balance()
{
	
	return 0;
}

//tools
void generate_inv_Gr_e(double e, double eta, GF &g)
{
	assert(g.size1_ == g.size2_);
	for(int w = 0; w < g.ngrid_; w++)
	{
		cplx ret = (g.grid_[w] - e + II * eta);
		for(int ind = 0; ind < g.size1_; ind++)
		{
			g.g_ret[w * g.el_size_ + ind * g.size2_ + ind] = ret;
		}
	}
}

template <class Matrix>
void generate_inv_Gr_e(const Matrix &e_mat, double eta, GF &g)
{
	assert(g.size1_ == g.size2_);
	assert(g.size1_ == e_mat.rows());
	
/*
	for(int w = 0; w < g.ngrid_; w++)
	{
		cdmatrix ret = (g.grid_[w] + II * eta) * Eigen::MatrixXd::Identity(e_mat.rows(), e_mat.cols()) - e_mat;
		for(int ind1 = 0; ind1 < g.size1_; ind1++)
		{
			for(int ind2 = 0; ind2 < g.size2_; ind2++)
			{
				g.g_ret[w * g.el_size_ + ind1 * g.size2_ + ind2] = ret(ind1,ind2);
			}
		}
	}
*/
	for(int w = 0; w < g.ngrid_; w++)
	{
		for(int ind1 = 0; ind1 < g.size1_; ind1++)
		{
			*(g.p_ret(w, ind1, ind1)) = g.grid_[w] + II * eta;
			for(int ind2 = 0; ind2 < g.size2_; ind2++)
			{
				*(g.p_ret(w, ind1, ind2)) -= e_mat(ind1, ind2);
			}
		}
	}
}

template <class Matrix>
void generate_inv_Gr(const Matrix &e_mat, GF &self, GF &g)
{
	assert(g.size1_ == g.size2_);
	assert(g.size1_ == self.size1_);
	assert(g.size2_ == self.size2_);
	assert(g.size1_ == e_mat.rows());
	assert(g.size2_ == e_mat.cols());
	
/*
	for(int w = 0; w < g.ngrid_; w++)
	{
		cdmatrix ret = (g.grid_[w] + II * eta) * Eigen::MatrixXd::Identity(e_mat.rows(), e_mat.cols()) - e_mat;
		for(int ind1 = 0; ind1 < g.size1_; ind1++)
		{
			for(int ind2 = 0; ind2 < g.size2_; ind2++)
			{
				g.g_ret[w * g.el_size_ + ind1 * g.size2_ + ind2] = ret(ind1,ind2);
			}
		}
	}
*/
	for(int w = 0; w < g.ngrid_; w++)
	{
		for(int ind1 = 0; ind1 < g.size1_; ind1++)
		{
			*(g.p_ret(w, ind1, ind1)) = g.grid_[w];
			for(int ind2 = 0; ind2 < g.size2_; ind2++)
			{
				*(g.p_ret(w, ind1, ind2)) -= e_mat(ind1, ind2) + *(self.p_ret(w, ind1, ind2));
			}
		}
	}
}

//const cdmatrix& GF::Greater(int w)
cplx GF::sGreater(int w) const
{
	//cdmatrix gtr(size1_, size2_);
	cplx gtr;
	#ifndef STAND_ALONE
	CNTR_ASSERT_LESEQ(NESS_ASSERT_0, w, ngrid_ - 1, __PRETTY_FUNCTION__);
	#else
	assert(w < ngrid_);
	#endif
	if(gf_type_ == freq_gf)
	{
		gtr = 2.0 * II * g_ret[w].imag() + g_les[w];
	}
	else if(gf_type_ == time_gf)
	{
		if(w > 0)
		{
			gtr = g_ret[w] + g_les[w];

		}
		else if(w == 0)
		{
			gtr = 2.0 * g_ret[w] + g_les[w];
		}
	}

	return gtr;
}

cdmatrix GF::Greater(int w) const
{
	cdmatrix gtr(size1_, size2_);
	assert(size1_ == size2_);
	assert(w < ngrid_);
	if(gf_type_ == freq_gf)
	{
		for(int idx1 = 0; idx1 < size1_; idx1++)
		{
			for(int idx2 = 0; idx2 < size2_; idx2++)
			{
				gtr(idx1, idx2) = g_ret[w * el_size_ + idx1 * size2_ + idx2].real() 
				- g_ret[w * el_size_ + idx2 * size2_ + idx1].real() +
				II * (g_ret[w * el_size_ + idx1 * size2_ + idx2].imag() 
				+ g_ret[w * el_size_ + idx2 * size2_ + idx1].imag()) 
				+ g_les[w * el_size_ + idx1 * size2_ + idx2];
			}
		}
	}
	else if(gf_type_ == time_gf)
	{
		#if FACTOR_TWO_MAGIC==1
        if(w > 0)
		{
			for(int idx1 = 0; idx1 < size1_; idx1++)
			{
				for(int idx2 = 0; idx2 < size2_; idx2++)
				{
					gtr(idx1, idx2) = g_ret[w * el_size_ + idx1 * size2_ + idx2] + g_les[w * el_size_ + idx1 * size2_ + idx2];
				}
			}

		}
		else if(w == 0)
		{
			for(int idx1 = 0; idx1 < size1_; idx1++)
			{
				for(int idx2 = 0; idx2 < size2_; idx2++)
				{
					gtr(idx1, idx2) = 2.0 * g_ret[w * el_size_ + idx1 * size2_ + idx2] + g_les[w * el_size_ + idx1 * size2_ + idx2];
				}
			}
		}
        #else // factor two magic
		{
			for(int idx1 = 0; idx1 < size1_; idx1++)
			{
				for(int idx2 = 0; idx2 < size2_; idx2++)
				{
					gtr(idx1, idx2) = g_ret[w * el_size_ + idx1 * size2_ + idx2] + g_les[w * el_size_ + idx1 * size2_ + idx2];
				}
			}

		}
        #endif // factor two magic
	}

	return gtr;
}

cdmatrix GF::pp_Greater(int w) const
{
	cdmatrix gtr(size1_, size2_);
	assert(size1_ == size2_);
	assert(w < ngrid_);
	if(gf_type_ == freq_gf)
	{
		for(int idx1 = 0; idx1 < size1_; idx1++)
		{
			for(int idx2 = 0; idx2 < size2_; idx2++)
			{
				gtr(idx1, idx2) = g_ret[w * el_size_ + idx1 * size2_ + idx2].real() 
				- g_ret[w * el_size_ + idx2 * size2_ + idx1].real() +
				II * (g_ret[w * el_size_ + idx1 * size2_ + idx2].imag() 
				+ g_ret[w * el_size_ + idx2 * size2_ + idx1].imag());
			}
		}
	}
	else if(gf_type_ == time_gf)
	{
		if(w > 0)
		{
			for(int idx1 = 0; idx1 < size1_; idx1++)
			{
				for(int idx2 = 0; idx2 < size2_; idx2++)
				{
					gtr(idx1, idx2) = g_ret[w * el_size_ + idx1 * size2_ + idx2];
				}
			}

		}
		else if(w == 0)
		{
			for(int idx1 = 0; idx1 < size1_; idx1++)
			{
				for(int idx2 = 0; idx2 < size2_; idx2++)
				{
					gtr(idx1, idx2) = 2.0 * g_ret[w * el_size_ + idx1 * size2_ + idx2];
				}
			}
		}
	}

	return gtr;
}

int GF::reset_grid(double dgrid)
{
	dgrid_ = dgrid;
	int halfn = ngrid_ / 2;
	if(gf_type_ == freq_gf)
	{
		for (int w_ind = 0; w_ind < halfn; w_ind++)
		{
			grid_[w_ind] = w_ind * dgrid_;
			grid_[ngrid_ - w_ind - 1] =  -(w_ind + 1) * dgrid_;
		}
	}
	else if(gf_type_ == time_gf)
	{
		for (int w_ind = 0; w_ind < ngrid_; w_ind++)
		{
			grid_[w_ind] = w_ind * dgrid_;
		}
	}

	return 0;
}

};//end of namespace
#endif
