/*
    Copyright (c) 2013, Philipp Krähenbühl
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
        * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
        * Neither the name of the Stanford University nor the
        names of its contributors may be used to endorse or promote products
        derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY Philipp Krähenbühl ''AS IS'' AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL Philipp Krähenbühl BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once
#include <cstdlib>
#include <vector>
#include <cstring>
#include <cassert>
#include <cstdio>
#include <cmath>
#include <Eigen/Core>
using namespace Eigen;

/************************************************/
/***          Permutohedral Lattice           ***/
/************************************************/

class Permutohedral
{
protected:
	struct Neighbors{
		int n1, n2;
		Neighbors( int n1=0, int n2=0 ):n1(n1),n2(n2){
		}
	};
	std::vector<int> offset_, rank_;
	std::vector<float> barycentric_;
	std::vector<Neighbors> blur_neighbors_;
	// Number of elements, size of sparse discretized space, dimension of features
	int N_, M_, d_;
	void sseCompute ( float* out, const float* in, int value_size, bool reverse=false ) const;
	void seqCompute ( float* out, const float* in, int value_size, bool reverse=false ) const;
public:
	Permutohedral();
	void init ( const MatrixXf & features );
	MatrixXf compute ( const MatrixXf & v, bool reverse=false ) const;
	void compute ( MatrixXf & out, const MatrixXf & in, bool reverse=false ) const;
	// Compute the gradient of a^T K b
	void gradient ( float* df, const float * a, const float* b, int value_size ) const;
};
