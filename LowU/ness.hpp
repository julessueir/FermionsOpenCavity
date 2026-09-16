#include "fftw3.h"

#define STAND_ALONE 

#ifndef STAND_ALONE
#define NO_IMPLEMENTATION
#include "cntr.hpp"
#undef NO_IMPLEMENTATION
#include "./ppsc/ppsc.hpp"
#endif

#include "ness_global_settings.hpp"
#include "ness_GF_decl.hpp"
#include "ness_fft_decl.hpp"
#include "ness_GF_impl.hpp"
#include "ness_fft_impl.hpp"
namespace ness
{
#ifndef STAND_ALONE
class ness_pseudo_particle_interaction
{
public:

  //typedef mam::static_matrix_type<1, 1> lam_matrix_type;
  //typedef ppsc::mam::dynamic_matrix_type lam_matrix_type;
  //typedef ppsc::cntr::herm_matrix_matrix_ref<lam_matrix_type> lam_ref_type;

  // scalar 
 ness_pseudo_particle_interaction(GF & lam) :
    lam(lam) {
    assert( lam.size1_ == 1 && lam.size2_ == 1 );
  }

  ness_pseudo_particle_interaction(GF & lam,
			      ppsc::operator_type op1,
			      ppsc::operator_type op2,
			      int sig=+1, // statistics +1 Boson, -1 Fermion
			      int dir=+1  // hybridization function direction fwd +1, bwd -1
			      ) :
    lam(lam), op1(op1), op2(op2), sig(sig), dir(dir), idx1(0), idx2(0) {
    assert( lam.size1_ == 1 && lam.size2_ == 1 );
  }

  ness_pseudo_particle_interaction(GF & lam,
			      int idx1, int idx2,
			      ppsc::operator_type op1,
			      ppsc::operator_type op2,
			      int sig=+1, // statistics +1 Boson, -1 Fermion
			      int dir=+1  // hybridization function direction fwd +1, bwd -1
			      ) :
    lam(lam), op1(op1), op2(op2), sig(sig), dir(dir), idx1(idx1), idx2(idx2) {}

  GF &lam;
  ppsc::operator_type op1, op2;
  int sig, dir;
  int idx1, idx2; // component indices lam(t, t')(idx1, idx2)
  
};
typedef ness_pseudo_particle_interaction pp_int_type;
typedef std::vector<pp_int_type> pp_ints_type;
#endif
}

#include "ness_utils.hpp"

#ifndef STAND_ALONE
#include "impurity.hpp"
#include "diag.hpp"
#include "nca.hpp"
#include "solver.hpp"
#endif
