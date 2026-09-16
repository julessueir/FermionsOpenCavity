



//Jiajun Li, Anne Matthies 2018
//issue: performance issue with eigen::map; should be optimized.
//issue: left/right multiply for GF's not tested.

#ifndef ness_GF
#define ness_GF

#include "ness_global_settings.hpp"

namespace ness{
enum {time_gf, freq_gf};

/*------ Green function class-------------*/
class GF {
	public:
	~GF();
	GF();
	GF(double , int , int , int , int gf_type);
	GF(double , int , int , int gf_type = freq_gf);
	GF(const GF& g);
	GF& operator=(const GF &g);
	GF operator+(const GF &g);
	int set_shiftGF(int shift);
	int incr(GF &gf, cplx weight=1.0);
	int smul(cplx weight);
	int clear();
	int left_multiply(GF &g, double weight=1.0);
	//template<class Matrix>
	int left_multiply(const cdmatrix& mat, double weight=1.0);
	int right_multiply(GF &g, double weight=1.0);
	//template<class Matrix>
	int right_multiply(const cdmatrix& mat, double weight=1.0);
	friend int dyson(GF &invg0, GF &self, GF &g);
	friend int dyson_from_inv(GF &g, GF &self);
	int gf_idx(int w) { return (gf_type_ == freq_gf ? w - ngrid_ / 2 : w); };
	int reset_grid(double dgrid);
	cdmatrix Greater(int) const;
	cdmatrix pp_Greater(int) const;
	cplx sGreater(int) const;

	double dgrid_;
	long ngrid_;
	int self_balance();
	int size1_, size2_;
	int el_size_, data_size_;
	int gf_type_;
	cplx *p_ret(int w_idx, int i, int j) { return g_ret + w_idx * el_size_ + i * size2_ + j; }
	cplx *p_les(int w_idx, int i, int j) { return g_les + w_idx * el_size_ + i * size2_ + j; }
    
    
    const cplx *p_ret(int w_idx, int i, int j) const { return g_ret + w_idx * el_size_ + i * size2_ + j; }
	const cplx *p_les(int w_idx, int i, int j) const { return g_les + w_idx * el_size_ + i * size2_ + j; }

	double* grid_;
	cplx *g_ret, *g_les;
	std::vector<Eigen::Map<Eigen::MatrixXcd, Eigen::RowMajor>> Retarded, Lesser;
};

typedef std::vector<GF> GFs;
};// end of the namespace
#endif

