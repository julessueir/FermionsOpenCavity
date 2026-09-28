#---------------------------------------------------------------------------#
# Compiler & Flags
#---------------------------------------------------------------------------#
CC := g++-14
CXXFLAGS := -std=c++17 -fpermissive -O3 -g -fopenmp

#---------------------------------------------------------------------------#
# Dependencies (Homebrew): fftw, eigen, boost
#   brew install fftw eigen boost
#---------------------------------------------------------------------------#
HOMEBREW_PREFIX ?= $(shell brew --prefix)
FFTW_INC        := -I$(HOMEBREW_PREFIX)/include
FFTW_LIB        := -L$(HOMEBREW_PREFIX)/lib -lfftw3

#---------------------------------------------------------------------------#
# Project includes & libs
#---------------------------------------------------------------------------#
INCS := -I/usr/include $(FFTW_INC)
LIBS := $(FFTW_LIB)

#---------------------------------------------------------------------------#
# ness_*.hpp/.cpp: vendored, standalone subset of the NESSi library used by
# A. Picano, J. Li and M. Eckstein, Phys. Rev. B 104, 085108 (2021) -- see
# the top-level README for the full reference.
#---------------------------------------------------------------------------#
NESS_SRCS    := ness.cpp
NESS_HEADERS := ness.hpp ness_utils.hpp \
                ness_fft_decl.hpp ness_fft_impl.hpp \
                ness_GF_decl.hpp ness_GF_impl.hpp \
                ness_global_settings.hpp
NESS_OBJ     := libness.o

#---------------------------------------------------------------------------#
# Default target (if you just run `make` it will build the first EXE)
#---------------------------------------------------------------------------#
# List all your executables here (optional)
EXES := $(patsubst %.cpp,%.ex,$(wildcard quantum_boltz_*.cpp))

all: $(EXES)

#---------------------------------------------------------------------------#
# Build libness.o
#---------------------------------------------------------------------------#
$(NESS_OBJ): $(NESS_SRCS) $(NESS_HEADERS)
	$(CC) $(CXXFLAGS) -c $< -o $@ $(INCS)

#---------------------------------------------------------------------------#
# Generic rule to build any quantum_boltz_*.ex from its .cpp + libness.o
#---------------------------------------------------------------------------#
%.ex: %.cpp $(NESS_OBJ)
	$(CC) $(CXXFLAGS) $< $(NESS_OBJ) -o $@ $(INCS) $(LIBS)

#---------------------------------------------------------------------------#
# Clean up
#---------------------------------------------------------------------------#
.PHONY: all clean
clean:
	rm -f *.o *.ex

