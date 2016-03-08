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

/**** LabelCompatibility models a function \mu(a,b) ****/
// To create your own label compatibility implement an "apply" function
// than computes out(a) = sum_{x_j} \mu( a, b ) Q(b) (where Q(b) is the mean-field
// marginal of a specific variable)
// See below for examples
class LabelCompatibility {
public:
	virtual ~LabelCompatibility();
	virtual void apply( MatrixXf & out, const MatrixXf & Q ) const = 0;
	// For non-symmetric pairwise potentials we would need to use the transpose of the pairwise term
	// for parameter learning
	virtual void applyTranspose( MatrixXf & out, const MatrixXf & Q ) const;
	
	// Training and parameters
	virtual VectorXf parameters() const;
	virtual void setParameters( const VectorXf & v );
	virtual VectorXf gradient( const MatrixXf & b, const MatrixXf & Q ) const;
};
/**** Implements potts \mu(a,b) = -w[a==b] ****/
class PottsCompatibility: public LabelCompatibility {
protected:
	float w_;
public:
	PottsCompatibility( float weight=1.0 );
	virtual void apply( MatrixXf & out_values, const MatrixXf & in_values ) const;
	
	// Training and parameters
	virtual VectorXf parameters() const;
	virtual void setParameters( const VectorXf & v );
	virtual VectorXf gradient( const MatrixXf & b, const MatrixXf & Q ) const;
};
/**** Implements diagonal \mu(a,b) = -[a==b]v(a) ****/
class DiagonalCompatibility: public LabelCompatibility {
protected:
	VectorXf w_;
public:
	DiagonalCompatibility( const VectorXf & v );
	virtual void apply( MatrixXf & out_values, const MatrixXf & in_values ) const;
	
	// Training and parameters
	virtual VectorXf parameters() const;
	virtual void setParameters( const VectorXf & v );
	virtual VectorXf gradient( const MatrixXf & b, const MatrixXf & Q ) const;
};
/**** Implements matrix \mu(a,b) [enforces symmetry, but not positive definitness] ****/
class MatrixCompatibility: public LabelCompatibility {
protected:
	MatrixXf w_;
public:
	MatrixCompatibility( const MatrixXf & m );
	virtual void apply( MatrixXf & out_values, const MatrixXf & in_values ) const;
	virtual void applyTranspose( MatrixXf & out_values, const MatrixXf & in_values ) const;
	
	// Training and parameters
	virtual VectorXf parameters() const;
	virtual void setParameters( const VectorXf & v );
	virtual VectorXf gradient( const MatrixXf & b, const MatrixXf & Q ) const;
};
