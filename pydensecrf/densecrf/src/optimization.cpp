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
#include "optimization.h"
#include "lbfgs.h"
#include <algorithm>
#include <cstdio>

static lbfgsfloatval_t evaluate(
    void *instance,
    const lbfgsfloatval_t *x,
    lbfgsfloatval_t *g,
    const int n,
    const lbfgsfloatval_t step
    )
{
	EnergyFunction * efun = static_cast<EnergyFunction*>( instance );
	
	VectorXf vx( n ), vg( n );
	std::copy( x, x+n, vx.data() );
	lbfgsfloatval_t r = efun->gradient( vx, vg );
	
	std::copy( vg.data(), vg.data()+n, g );
	return r;
}

static int progress(
    void *instance,
    const lbfgsfloatval_t *x,
    const lbfgsfloatval_t *g,
    const lbfgsfloatval_t fx,
    const lbfgsfloatval_t xnorm,
    const lbfgsfloatval_t gnorm,
    const lbfgsfloatval_t step,
    int n,
    int k,
    int ls
    )
{
    printf("Iteration %d:\n", k);
    printf("  fx = %f, xnorm = %f, gnorm = %f, step = %f\n", fx, xnorm, gnorm, step);
    printf("\n");
    return 0;
}
VectorXf minimizeLBFGS( EnergyFunction & efun, int restart, bool verbose ) {
	VectorXf x0 = efun.initialValue();
	const int n = x0.rows();
	
	lbfgsfloatval_t *x = lbfgs_malloc(n);
	if (x == NULL) {
		printf("ERROR: Failed to allocate a memory block for variables.\n");
		return x0;
	}
	std::copy( x0.data(), x0.data()+n, x );
	
	lbfgs_parameter_t param;
	lbfgs_parameter_init(&param);
	// You might want to adjust the parameters to your problem
	param.epsilon = 1e-6;
	param.max_iterations = 50;
	
	double last_f = 1e100;
	int ret;
	for( int i=0; i<=restart; i++ ) {
		lbfgsfloatval_t fx;
		ret = lbfgs(n, x, &fx, evaluate, verbose?progress:NULL, &efun, &param);
		if( last_f > fx )
			last_f = fx;
		else
			break;
	}
	
	if ( verbose ) {
		printf("L-BFGS optimization terminated with status code = %d\n", ret);
	}
	
	std::copy( x, x+n, x0.data() );
	lbfgs_free(x);
	return x0;
}
VectorXf numericGradient( EnergyFunction & efun, const VectorXf & x, float EPS ) {
	VectorXf g( x.rows() ), tmp;
	for( int i=0; i<x.rows(); i++ ) {
		VectorXf xx = x;
		xx[i] = x[i]+EPS;
		double v1 = efun.gradient( xx, tmp );
		xx[i] = x[i]-EPS;
		double v0 = efun.gradient( xx, tmp );
		g[i] = (v1-v0)/(2*EPS);
	}
	return g;
}
VectorXf gradient( EnergyFunction & efun, const VectorXf & x ) {
	VectorXf r( x.rows() );
	efun.gradient( x, r );
	return r;
}
double gradCheck( EnergyFunction & efun, const VectorXf & x, float EPS ) {
	VectorXf ng = numericGradient( efun, x, EPS );
	VectorXf g( x.rows() );
	efun.gradient( x, g );
	return (ng-g).norm();
}

VectorXf computeFunction( EnergyFunction & efun, const VectorXf & x, const VectorXf & dx, int n_samples ) {
	VectorXf r( n_samples );
	VectorXf tmp = x;
	for( int i=0; i<n_samples; i++ )
		r[i] = efun.gradient( x+i*dx, tmp );
	return r;
}

