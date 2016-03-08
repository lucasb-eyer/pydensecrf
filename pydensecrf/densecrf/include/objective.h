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
#include <Eigen/Core>
using namespace Eigen;

typedef Matrix<short,Dynamic,1> VectorXs;

/**** Learning Objective ****/
class ObjectiveFunction {
public:
	virtual ~ObjectiveFunction();
	// Evaluate an objective function L(Q) and its gradient \nabla L(Q)
	// Return the objetive value L(Q) and set gradient[i*M+l] to Q_i(l)*\partial L / \partial Q_i(l)
	// We use the scales gradient here for numerical reasons!
	virtual double evaluate( MatrixXf & d_mul_Q, const MatrixXf & Q ) const = 0;
};
// Log likelihood objective
class LogLikelihood: public ObjectiveFunction {
protected:
	VectorXs gt_;
	float robust_;
public:
	// Give a ground_truth labeling of size N, optional use a robustness term robust>0
	LogLikelihood( const VectorXs & gt, float robust=0 );
	// The objective value is sum_i log( Q_i( ground_truth_i ) + robust )
	virtual double evaluate( MatrixXf & d_mul_Q, const MatrixXf & Q ) const;
};
// Log likelihood objective
class Hamming: public ObjectiveFunction {
protected:
	VectorXs gt_;
	VectorXf class_weight_;
public:
	// Give a ground_truth labeling of size N, reweight classes to cope with an invariance
	// weight by w_c = pow( #labels_c, -class_weight_pow )
	Hamming( const VectorXs & gt, float class_weight_pow=0 );
	Hamming( const VectorXs & gt, const VectorXf & class_weight );
	// The objective value is sum_i Q_i( ground_truth_i )
	virtual double evaluate( MatrixXf & d_mul_Q, const MatrixXf & Q ) const;
};
// Intersection over union objective
class IntersectionOverUnion: public ObjectiveFunction {
protected:
	VectorXs gt_;
public:
	// Give a ground_truth labeling of size N
	IntersectionOverUnion( const VectorXs & gt );
	// The objective value is sum_l ( sum_i [ground_truth_i == l] Q_i( l ) ) / ( |ground_truth_i == l| + sum_i [ground_truth_i != l] Q_i( l ) )
	virtual double evaluate( MatrixXf & d_mul_Q, const MatrixXf & Q ) const;
};
