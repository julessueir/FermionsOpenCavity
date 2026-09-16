#ifndef GLOBAL
#define GLOBAL
#include <iostream>
#include <complex>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstring>
#ifndef STAND_ALONE
	#include "./ppsc/ppsc.hpp"
#else
	#include "eigen3/Eigen/Dense"
#endif

namespace ness{
#define NESS_ASSERT_0 1
typedef std::complex<double> cplx;

#ifndef STAND_ALONE
using ppsc::gf_vert_type;
using ppsc::gf_verts_type;
#else
using Eigen::MatrixXcd;
typedef MatrixXcd cdmatrix;
const cplx II(0.0, 1.0);
#endif

#define FACTOR_TWO_MAGIC 1

} //end of namespace
#endif
