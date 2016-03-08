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
#include "unary.h"


UnaryEnergy::~UnaryEnergy() {
}
VectorXf UnaryEnergy::parameters() const {
	return VectorXf();
}
void UnaryEnergy::setParameters( const VectorXf & v ) {
}
VectorXf UnaryEnergy::gradient( const MatrixXf & b ) const {
	return VectorXf();
}


ConstUnaryEnergy::ConstUnaryEnergy( const MatrixXf & u ):unary_(u) {
}
MatrixXf ConstUnaryEnergy::get() const {
	return unary_;
}

LogisticUnaryEnergy::LogisticUnaryEnergy( const MatrixXf & L, const MatrixXf & f ):L_(L),f_(f) {
}
MatrixXf LogisticUnaryEnergy::get() const {
	return L_*f_;
}
VectorXf LogisticUnaryEnergy::parameters() const {
	MatrixXf l = L_;
	l.resize( l.cols()*l.rows(), 1 );
	return l;
}
void LogisticUnaryEnergy::setParameters( const VectorXf & v ) {
	assert( v.rows() == L_.cols()*L_.rows() );
	MatrixXf l = v;
	l.resizeLike( L_ );
	L_ = l;
}
VectorXf LogisticUnaryEnergy::gradient( const MatrixXf & b ) const {
	MatrixXf g = b*f_.transpose();
	g.resize( g.cols()*g.rows(), 1 );
	return g;
}
