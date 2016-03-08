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
#include "pairwise.h"
#include <iostream>

Kernel::~Kernel() {
}
class DenseKernel: public Kernel {
protected:
	NormalizationType ntype_;
	KernelType ktype_;
	Permutohedral lattice_;
	VectorXf norm_;
	MatrixXf f_;
	MatrixXf parameters_;
	void initLattice( const MatrixXf & f ) {
		const int N = f.cols();
		lattice_.init( f );
		
		norm_ = lattice_.compute( VectorXf::Ones( N ).transpose() ).transpose();
		
		if ( ntype_ == NO_NORMALIZATION ) {
			float mean_norm = 0;
			for ( int i=0; i<N; i++ )
				mean_norm += norm_[i];
			mean_norm = N / mean_norm;
			for ( int i=0; i<N; i++ )
				norm_[i] = mean_norm;
		}
		else if ( ntype_ == NORMALIZE_SYMMETRIC ) {
			for ( int i=0; i<N; i++ )
				norm_[i] = 1.0 / sqrt(norm_[i]+1e-20);
		}
		else {
			for ( int i=0; i<N; i++ )
				norm_[i] = 1.0 / (norm_[i]+1e-20);
		}
	}
	void filter( MatrixXf & out, const MatrixXf & in, bool transpose ) const {
		// Read in the values
		if( ntype_ == NORMALIZE_SYMMETRIC || (ntype_ == NORMALIZE_BEFORE && !transpose) || (ntype_ == NORMALIZE_AFTER && transpose))
			out = in*norm_.asDiagonal();
		else
			out = in;
	
		// Filter
		if( transpose )
			lattice_.compute( out, out, true );
		else
			lattice_.compute( out, out );
// 			lattice_.compute( out.data(), out.data(), out.rows() );
	
		// Normalize again
		if( ntype_ == NORMALIZE_SYMMETRIC || (ntype_ == NORMALIZE_BEFORE && transpose) || (ntype_ == NORMALIZE_AFTER && !transpose))
			out = out*norm_.asDiagonal();
	}
	// Compute d/df a^T*K*b
	MatrixXf kernelGradient( const MatrixXf & a, const MatrixXf & b ) const {
		MatrixXf g = 0*f_;
		lattice_.gradient( g.data(), a.data(), b.data(), a.rows() );
		return g;
	}
	MatrixXf featureGradient( const MatrixXf & a, const MatrixXf & b ) const {
		if (ntype_ == NO_NORMALIZATION )
			return kernelGradient( a, b );
		else if (ntype_ == NORMALIZE_SYMMETRIC ) {
			MatrixXf fa = lattice_.compute( a*norm_.asDiagonal(), true );
			MatrixXf fb = lattice_.compute( b*norm_.asDiagonal() );
			MatrixXf ones = MatrixXf::Ones( a.rows(), a.cols() );
			VectorXf norm3 = norm_.array()*norm_.array()*norm_.array();
			MatrixXf r = kernelGradient( 0.5*( a.array()*fb.array() + fa.array()*b.array() ).matrix()*norm3.asDiagonal(), ones );
			return - r + kernelGradient( a*norm_.asDiagonal(), b*norm_.asDiagonal() );
		}
		else if (ntype_ == NORMALIZE_AFTER ) {
			MatrixXf fb = lattice_.compute( b );
			
			MatrixXf ones = MatrixXf::Ones( a.rows(), a.cols() );
			VectorXf norm2 = norm_.array()*norm_.array();
			MatrixXf r = kernelGradient( ( a.array()*fb.array() ).matrix()*norm2.asDiagonal(), ones );
			return - r + kernelGradient( a*norm_.asDiagonal(), b );
		}
		else /*if (ntype_ == NORMALIZE_BEFORE )*/ {
			MatrixXf fa = lattice_.compute( a, true );
			
			MatrixXf ones = MatrixXf::Ones( a.rows(), a.cols() );
			VectorXf norm2 = norm_.array()*norm_.array();
			MatrixXf r = kernelGradient( ( fa.array()*b.array() ).matrix()*norm2.asDiagonal(), ones );
			return -r+kernelGradient( a, b*norm_.asDiagonal() );
		}
	}
public:
	DenseKernel(const MatrixXf & f, KernelType ktype, NormalizationType ntype):f_(f), ktype_(ktype), ntype_(ntype) {
		if (ktype_ == DIAG_KERNEL)
			parameters_ = VectorXf::Ones( f.rows() );
		else if( ktype == FULL_KERNEL )
			parameters_ = MatrixXf::Identity( f.rows(), f.rows() );
		initLattice( f );
	}
	virtual void apply( MatrixXf & out, const MatrixXf & Q ) const {
		filter( out, Q, false );
	}
	virtual void applyTranspose( MatrixXf & out, const MatrixXf & Q ) const {
		filter( out, Q, true );
	}
	virtual VectorXf parameters() const {
		if (ktype_ == CONST_KERNEL)
			return VectorXf();
		else if (ktype_ == DIAG_KERNEL)
			return parameters_;
		else {
			MatrixXf p = parameters_;
			p.resize( p.cols()*p.rows(), 1 );
			return p;
		}
	}
	virtual void setParameters( const VectorXf & p ) {
		if (ktype_ == DIAG_KERNEL) {
			parameters_ = p;
			initLattice( p.asDiagonal() * f_ );
		}
		else if (ktype_ == FULL_KERNEL) {
			MatrixXf tmp = p;
			tmp.resize( parameters_.rows(), parameters_.cols() );
			parameters_ = tmp;
			initLattice( tmp * f_ );
		}
	}
	virtual VectorXf gradient( const MatrixXf & a, const MatrixXf & b ) const {
		if (ktype_ == CONST_KERNEL)
			return VectorXf();
		MatrixXf fg = featureGradient( a, b );
		if (ktype_ == DIAG_KERNEL)
			return (f_.array()*fg.array()).rowwise().sum();
		else {
			MatrixXf p = fg*f_.transpose();
			p.resize( p.cols()*p.rows(), 1 );
			return p;
		}
	}
};

PairwisePotential::~PairwisePotential(){
	delete compatibility_;
	delete kernel_;
}
PairwisePotential::PairwisePotential(const MatrixXf & features, LabelCompatibility * compatibility, KernelType ktype, NormalizationType ntype) : compatibility_(compatibility) {
	kernel_ = new DenseKernel( features, ktype, ntype );
}
void PairwisePotential::apply(MatrixXf & out, const MatrixXf & Q) const {
	kernel_->apply( out, Q );
	
	// Apply the compatibility
	compatibility_->apply( out, out );
}
void PairwisePotential::applyTranspose(MatrixXf & out, const MatrixXf & Q) const {
	kernel_->applyTranspose( out, Q );
	// Apply the compatibility
	compatibility_->applyTranspose( out, out );
}
VectorXf PairwisePotential::parameters() const {
	return compatibility_->parameters();
}
void PairwisePotential::setParameters( const VectorXf & v ) {
	compatibility_->setParameters( v );
}
VectorXf PairwisePotential::gradient( const MatrixXf & b, const MatrixXf & Q ) const {
	MatrixXf filtered_Q = 0*Q;
	// You could reuse the filtered_b from applyTranspose
	kernel_->apply( filtered_Q, Q );
	return compatibility_->gradient(b,filtered_Q);
}
VectorXf PairwisePotential::kernelParameters() const {
	return kernel_->parameters();
}
void PairwisePotential::setKernelParameters( const VectorXf & v ) {
	kernel_->setParameters( v );
}
VectorXf PairwisePotential::kernelGradient( const MatrixXf & b, const MatrixXf & Q ) const {
	MatrixXf lbl_Q = 0*Q;
	// You could reuse the filtered_b from applyTranspose
	compatibility_->apply( lbl_Q, Q );
	return kernel_->gradient(b,lbl_Q);
}