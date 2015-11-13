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
#include "labelcompatibility.h"

LabelCompatibility::~LabelCompatibility() {
}
void LabelCompatibility::applyTranspose( MatrixXf & out, const MatrixXf & Q ) const {
	apply( out, Q );
}
VectorXf LabelCompatibility::parameters() const {
	return VectorXf();
}
void LabelCompatibility::setParameters( const VectorXf & v ) {
}
VectorXf LabelCompatibility::gradient( const MatrixXf & b, const MatrixXf & Q ) const {
	return VectorXf();
}


PottsCompatibility::PottsCompatibility( float weight ): w_(weight) {
}
void PottsCompatibility::apply( MatrixXf & out, const MatrixXf & Q ) const {
	out = -w_*Q;
}
VectorXf PottsCompatibility::parameters() const {
	VectorXf r(1);
	r[0] = w_;
	return r;
}
void PottsCompatibility::setParameters( const VectorXf & v ) {
	w_ = v[0];
}
VectorXf PottsCompatibility::gradient( const MatrixXf & b, const MatrixXf & Q ) const {
	VectorXf r(1);
	r[0] = -(b.array()*Q.array()).sum();
	return r;
}


DiagonalCompatibility::DiagonalCompatibility( const VectorXf & v ): w_(v) {
}
void DiagonalCompatibility::apply( MatrixXf & out, const MatrixXf & Q ) const {
	assert( w_.rows() == Q.rows() );
	out = w_.asDiagonal()*Q;
}
VectorXf DiagonalCompatibility::parameters() const {
	return w_;
}
void DiagonalCompatibility::setParameters( const VectorXf & v ) {
	w_ = v;
}
VectorXf DiagonalCompatibility::gradient( const MatrixXf & b, const MatrixXf & Q ) const {
	return (b.array()*Q.array()).rowwise().sum();
}
MatrixCompatibility::MatrixCompatibility( const MatrixXf & m ): w_(0.5*(m + m.transpose())) {
	assert( m.cols() == m.rows() );
}
void MatrixCompatibility::apply( MatrixXf & out, const MatrixXf & Q ) const {
	out = w_*Q;
}
void MatrixCompatibility::applyTranspose( MatrixXf & out, const MatrixXf & Q ) const {
	out = w_.transpose()*Q;
}
VectorXf MatrixCompatibility::parameters() const {
	VectorXf r( w_.cols()*(w_.rows()+1)/2 );
	for( int i=0,k=0; i<w_.cols(); i++ )
		for( int j=i; j<w_.rows(); j++, k++ )
			r[k] = w_(i,j);
	return r;
}
void MatrixCompatibility::setParameters( const VectorXf & v ) {
	assert( v.rows() == w_.cols()*(w_.rows()+1)/2 );
	for( int i=0,k=0; i<w_.cols(); i++ )
		for( int j=i; j<w_.rows(); j++, k++ )
			w_(j,i) = w_(i,j) = v[k];
}
VectorXf MatrixCompatibility::gradient( const MatrixXf & b, const MatrixXf & Q ) const {
	MatrixXf g = b * Q.transpose();
	VectorXf r( w_.cols()*(w_.rows()+1)/2 );
	for( int i=0,k=0; i<g.cols(); i++ )
		for( int j=i; j<g.rows(); j++, k++ )
			r[k] = g(i,j) + (i!=j?g(j,i):0.f);
	return r;
}
	