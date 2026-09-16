
#define STAND_ALONE 
#include <fftw3.h>
#include <eigen3/Eigen/Dense>
#include "ness_global_settings.hpp"
#include "ness_GF_decl.hpp"
#include "ness_fft_decl.hpp"
namespace ness
{
void force_FD(GF &gf, double temp);

void force_imag(GF &gf);

int set_bwd_from_fwd(GF & gf_bwd, GF & gf_fwd);

int dyson(GF &invg0, GF &self, GF &g);


int dyson_from_inv(GF &g, GF &self);

int invert_invG(GF &g, GF &self);

double GF2norm(GF &g1, GF &g2);

void outputGF(GF &G, std::string filename);
void outputGF_new(GF &G, std::string filename);
void outputGF_new_F(GF &G, GF &F, std::string filename);
void outputGF_reduced(GF &G, std::string filename);

int healthCheck(GF &G);
};//end of namespace
