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

#include "permutohedral.h"

//#ifdef WIN32
//inline int round(double X) {
//	return int(X+.5);
//}
//#endif

#ifdef __SSE__
// SSE Permutoheral lattice
# define SSE_PERMUTOHEDRAL
#endif

#if defined(SSE_PERMUTOHEDRAL)
# include <emmintrin.h>
# include <xmmintrin.h>
# ifdef __SSE4_1__
#  include <smmintrin.h>
# endif
#endif


/************************************************/
/***                Hash Table                ***/
/************************************************/

class HashTable{
protected:
	size_t key_size_, filled_, capacity_;
	std::vector< short > keys_;
	std::vector< int > table_;
	void grow(){
		// Create the new memory and copy the values in
		int old_capacity = capacity_;
		capacity_ *= 2;
		std::vector<short> old_keys( (old_capacity+10)*key_size_ );
		std::copy( keys_.begin(), keys_.end(), old_keys.begin() );
		std::vector<int> old_table( capacity_, -1 );
		
		// Swap the memory
		table_.swap( old_table );
		keys_.swap( old_keys );
		
		// Reinsert each element
		for( int i=0; i<old_capacity; i++ )
			if (old_table[i] >= 0){
				int e = old_table[i];
				size_t h = hash( getKey(e) ) % capacity_;
				for(; table_[h] >= 0; h = h<capacity_-1 ? h+1 : 0);
				table_[h] = e;
			}
	}
	size_t hash( const short * k ) {
		size_t r = 0;
		for( size_t i=0; i<key_size_; i++ ){
			r += k[i];
			r *= 1664525;
		}
		return r;
	}
public:
	explicit HashTable( int key_size, int n_elements ) : key_size_ ( key_size ), filled_(0), capacity_(2*n_elements), keys_((capacity_/2+10)*key_size_), table_(2*n_elements,-1) {
	}
	int size() const {
		return filled_;
	}
	void reset() {
		filled_ = 0;
		std::fill( table_.begin(), table_.end(), -1 );
	}
	int find( const short * k, bool create = false ){
		if (2*filled_ >= capacity_) grow();
		// Get the hash value
		size_t h = hash( k ) % capacity_;
		// Find the element with he right key, using linear probing
		while(1){
			int e = table_[h];
			if (e==-1){
				if (create){
					// Insert a new key and return the new id
					for( size_t i=0; i<key_size_; i++ )
						keys_[ filled_*key_size_+i ] = k[i];
					return table_[h] = filled_++;
				}
				else
					return -1;
			}
			// Check if the current key is The One
			bool good = true;
			for( size_t i=0; i<key_size_ && good; i++ )
				if (keys_[ e*key_size_+i ] != k[i])
					good = false;
			if (good)
				return e;
			// Continue searching
			h++;
			if (h==capacity_) h = 0;
		}
	}
	const short * getKey( int i ) const{
		return &keys_[i*key_size_];
	}

};

/************************************************/
/***          Permutohedral Lattice           ***/
/************************************************/

Permutohedral::Permutohedral():N_( 0 ), M_( 0 ), d_( 0 ) {
}
#ifdef SSE_PERMUTOHEDRAL
void Permutohedral::init ( const MatrixXf & feature )
{
	// Compute the lattice coordinates for each feature [there is going to be a lot of magic here
	N_ = feature.cols();
	d_ = feature.rows();
	HashTable hash_table( d_, N_/**(d_+1)*/ );
	
	const int blocksize = sizeof(__m128) / sizeof(float);
	const __m128 invdplus1   = _mm_set1_ps( 1.0f / (d_+1) );
	const __m128 dplus1      = _mm_set1_ps( d_+1 );
	const __m128 Zero        = _mm_set1_ps( 0 );
	const __m128 One         = _mm_set1_ps( 1 );

	// Allocate the class memory
	offset_.resize( (d_+1)*(N_+16) );
	std::fill( offset_.begin(), offset_.end(), 0 );
	barycentric_.resize( (d_+1)*(N_+16) );
	std::fill( barycentric_.begin(), barycentric_.end(), 0 );
	rank_.resize( (d_+1)*(N_+16) );
	
	// Allocate the local memory
	__m128 * scale_factor = (__m128*) _mm_malloc( (d_  )*sizeof(__m128) , 16 );
	__m128 * f            = (__m128*) _mm_malloc( (d_  )*sizeof(__m128) , 16 );
	__m128 * elevated     = (__m128*) _mm_malloc( (d_+1)*sizeof(__m128) , 16 );
	__m128 * rem0         = (__m128*) _mm_malloc( (d_+1)*sizeof(__m128) , 16 );
	__m128 * rank         = (__m128*) _mm_malloc( (d_+1)*sizeof(__m128), 16 );
	float * barycentric = new float[(d_+2)*blocksize];
	short * canonical = new short[(d_+1)*(d_+1)];
	short * key = new short[d_+1];
	
	// Compute the canonical simplex
	for( int i=0; i<=d_; i++ ){
		for( int j=0; j<=d_-i; j++ )
			canonical[i*(d_+1)+j] = i;
		for( int j=d_-i+1; j<=d_; j++ )
			canonical[i*(d_+1)+j] = i - (d_+1);
	}
	
	// Expected standard deviation of our filter (p.6 in [Adams etal 2010])
	float inv_std_dev = sqrt(2.0 / 3.0)*(d_+1);
	// Compute the diagonal part of E (p.5 in [Adams etal 2010])
	for( int i=0; i<d_; i++ )
		scale_factor[i] = _mm_set1_ps( 1.0 / sqrt( (i+2)*(i+1) ) * inv_std_dev );
	
	// Setup the SSE rounding
#ifndef __SSE4_1__
	const unsigned int old_rounding = _mm_getcsr();
	_mm_setcsr( (old_rounding&~_MM_ROUND_MASK) | _MM_ROUND_NEAREST );
#endif

	// Compute the simplex each feature lies in
	for( int k=0; k<N_; k+=blocksize ){
		// Load the feature from memory
		float * ff = (float*)f;
		for( int j=0; j<d_; j++ )
			for( int i=0; i<blocksize; i++ )
				ff[ j*blocksize + i ] = k+i < N_ ? feature(j,k+i) : 0.0;
		
		// Elevate the feature ( y = Ep, see p.5 in [Adams etal 2010])
		
		// sm contains the sum of 1..n of our faeture vector
		__m128 sm = Zero;
		for( int j=d_; j>0; j-- ){
			__m128 cf = f[j-1]*scale_factor[j-1];
			elevated[j] = sm - _mm_set1_ps(j)*cf;
			sm += cf;
		}
		elevated[0] = sm;
		
		// Find the closest 0-colored simplex through rounding
		__m128 sum = Zero;
		for( int i=0; i<=d_; i++ ){
			__m128 v = invdplus1 * elevated[i];
#ifdef __SSE4_1__
			v = _mm_round_ps( v, _MM_FROUND_TO_NEAREST_INT );
#else
			v = _mm_cvtepi32_ps( _mm_cvtps_epi32( v ) );
#endif
			rem0[i] = v*dplus1;
			sum += v;
		}
		
		// Find the simplex we are in and store it in rank (where rank describes what position coorinate i has in the sorted order of the features values)
		for( int i=0; i<=d_; i++ )
			rank[i] = Zero;
		for( int i=0; i<d_; i++ ){
			__m128 di = elevated[i] - rem0[i];
			for( int j=i+1; j<=d_; j++ ){
				__m128 dj = elevated[j] - rem0[j];
				__m128 c = _mm_and_ps( One, _mm_cmplt_ps( di, dj ) );
				rank[i] += c;
				rank[j] += One-c;
			}
		}
		
		// If the point doesn't lie on the plane (sum != 0) bring it back
		for( int i=0; i<=d_; i++ ){
			rank[i] += sum;
			__m128 add = _mm_and_ps( dplus1, _mm_cmplt_ps( rank[i], Zero ) );
			__m128 sub = _mm_and_ps( dplus1, _mm_cmpge_ps( rank[i], dplus1 ) );
			rank[i] += add-sub;
			rem0[i] += add-sub;
		}
		
		// Compute the barycentric coordinates (p.10 in [Adams etal 2010])
		for( int i=0; i<(d_+2)*blocksize; i++ )
			barycentric[ i ] = 0;
		for( int i=0; i<=d_; i++ ){
			__m128 v = (elevated[i] - rem0[i])*invdplus1;
			
			// Didn't figure out how to SSE this
			float * fv = (float*)&v;
			float * frank = (float*)&rank[i];
			for( int j=0; j<blocksize; j++ ){
				int p = d_-frank[j];
				barycentric[j*(d_+2)+p  ] += fv[j];
				barycentric[j*(d_+2)+p+1] -= fv[j];
			}
		}
		
		// The rest is not SSE'd
		for( int j=0; j<blocksize; j++ ){
			// Wrap around
			barycentric[j*(d_+2)+0]+= 1 + barycentric[j*(d_+2)+d_+1];
			
			float * frank = (float*)rank;
			float * frem0 = (float*)rem0;
			// Compute all vertices and their offset
			for( int remainder=0; remainder<=d_; remainder++ ){
				for( int i=0; i<d_; i++ ){
					key[i] = frem0[i*blocksize+j] + canonical[ remainder*(d_+1) + (int)frank[i*blocksize+j] ];
				}
				offset_[ (j+k)*(d_+1)+remainder ] = hash_table.find( key, true );
				rank_[ (j+k)*(d_+1)+remainder ] = frank[remainder*blocksize+j];
				barycentric_[ (j+k)*(d_+1)+remainder ] = barycentric[ j*(d_+2)+remainder ];
			}
		}
	}
	_mm_free( scale_factor );
	_mm_free( f );
	_mm_free( elevated );
	_mm_free( rem0 );
	_mm_free( rank );
	delete [] barycentric;
	delete [] canonical;
	delete [] key;
	
	// Reset the SSE rounding
#ifndef __SSE4_1__
	_mm_setcsr( old_rounding );
#endif
	
	// This is normally fast enough so no SSE needed here
	// Find the Neighbors of each lattice point
	
	// Get the number of vertices in the lattice
	M_ = hash_table.size();
	
	// Create the neighborhood structure
	blur_neighbors_.resize( (d_+1)*M_ );
	
	short * n1 = new short[d_+1];
	short * n2 = new short[d_+1];
	
	// For each of d+1 axes,
	for( int j = 0; j <= d_; j++ ){
		for( int i=0; i<M_; i++ ){
			const short * key = hash_table.getKey( i );
			for( int k=0; k<d_; k++ ){
				n1[k] = key[k] - 1;
				n2[k] = key[k] + 1;
			}
			n1[j] = key[j] + d_;
			n2[j] = key[j] - d_;
			
			blur_neighbors_[j*M_+i].n1 = hash_table.find( n1 );
			blur_neighbors_[j*M_+i].n2 = hash_table.find( n2 );
		}
	}
	delete[] n1;
	delete[] n2;
}
#else
void Permutohedral::init ( const MatrixXf & feature )
{
	// Compute the lattice coordinates for each feature [there is going to be a lot of magic here
	N_ = feature.cols();
	d_ = feature.rows();
	HashTable hash_table( d_, N_*(d_+1) );

	// Allocate the class memory
	offset_.resize( (d_+1)*N_ );
	rank_.resize( (d_+1)*N_ );
	barycentric_.resize( (d_+1)*N_ );
	
	// Allocate the local memory
	float * scale_factor = new float[d_];
	float * elevated = new float[d_+1];
	float * rem0 = new float[d_+1];
	float * barycentric = new float[d_+2];
	short * rank = new short[d_+1];
	short * canonical = new short[(d_+1)*(d_+1)];
	short * key = new short[d_+1];
	
	// Compute the canonical simplex
	for( int i=0; i<=d_; i++ ){
		for( int j=0; j<=d_-i; j++ )
			canonical[i*(d_+1)+j] = i;
		for( int j=d_-i+1; j<=d_; j++ )
			canonical[i*(d_+1)+j] = i - (d_+1);
	}
	
	// Expected standard deviation of our filter (p.6 in [Adams etal 2010])
	float inv_std_dev = sqrt(2.0 / 3.0)*(d_+1);
	// Compute the diagonal part of E (p.5 in [Adams etal 2010])
	for( int i=0; i<d_; i++ )
		scale_factor[i] = 1.0 / sqrt( double((i+2)*(i+1)) ) * inv_std_dev;
	
	// Compute the simplex each feature lies in
	for( int k=0; k<N_; k++ ){
		// Elevate the feature ( y = Ep, see p.5 in [Adams etal 2010])
		const float * f = &feature(0,k);
		
		// sm contains the sum of 1..n of our faeture vector
		float sm = 0;
		for( int j=d_; j>0; j-- ){
			float cf = f[j-1]*scale_factor[j-1];
			elevated[j] = sm - j*cf;
			sm += cf;
		}
		elevated[0] = sm;
		
		// Find the closest 0-colored simplex through rounding
		float down_factor = 1.0f / (d_+1);
		float up_factor = (d_+1);
		int sum = 0;
		for( int i=0; i<=d_; i++ ){
			//int rd1 = round( down_factor * elevated[i]);
			int rd2;
			float v = down_factor * elevated[i];
			float up = ceilf(v)*up_factor;
			float down = floorf(v)*up_factor;
			if (up - elevated[i] < elevated[i] - down) rd2 = (short)up;
			else rd2 = (short)down;

			//if(rd1!=rd2)
			//	break;

			rem0[i] = rd2;
			sum += rd2*down_factor;
		}
		
		// Find the simplex we are in and store it in rank (where rank describes what position coorinate i has in the sorted order of the features values)
		for( int i=0; i<=d_; i++ )
			rank[i] = 0;
		for( int i=0; i<d_; i++ ){
			double di = elevated[i] - rem0[i];
			for( int j=i+1; j<=d_; j++ )
				if ( di < elevated[j] - rem0[j])
					rank[i]++;
				else
					rank[j]++;
		}
		
		// If the point doesn't lie on the plane (sum != 0) bring it back
		for( int i=0; i<=d_; i++ ){
			rank[i] += sum;
			if ( rank[i] < 0 ){
				rank[i] += d_+1;
				rem0[i] += d_+1;
			}
			else if ( rank[i] > d_ ){
				rank[i] -= d_+1;
				rem0[i] -= d_+1;
			}
		}
		
		// Compute the barycentric coordinates (p.10 in [Adams etal 2010])
		for( int i=0; i<=d_+1; i++ )
			barycentric[i] = 0;
		for( int i=0; i<=d_; i++ ){
			float v = (elevated[i] - rem0[i])*down_factor;
			barycentric[d_-rank[i]  ] += v;
			barycentric[d_-rank[i]+1] -= v;
		}
		// Wrap around
		barycentric[0] += 1.0 + barycentric[d_+1];
		
		// Compute all vertices and their offset
		for( int remainder=0; remainder<=d_; remainder++ ){
			for( int i=0; i<d_; i++ )
				key[i] = rem0[i] + canonical[ remainder*(d_+1) + rank[i] ];
			offset_[ k*(d_+1)+remainder ] = hash_table.find( key, true );
			rank_[ k*(d_+1)+remainder ] = rank[remainder];
			barycentric_[ k*(d_+1)+remainder ] = barycentric[ remainder ];
		}
	}
	delete [] scale_factor;
	delete [] elevated;
	delete [] rem0;
	delete [] barycentric;
	delete [] rank;
	delete [] canonical;
	delete [] key;
	
	
	// Find the Neighbors of each lattice point
	
	// Get the number of vertices in the lattice
	M_ = hash_table.size();
	
	// Create the neighborhood structure
	blur_neighbors_.resize( (d_+1)*M_ );
	
	short * n1 = new short[d_+1];
	short * n2 = new short[d_+1];
	
	// For each of d+1 axes,
	for( int j = 0; j <= d_; j++ ){
		for( int i=0; i<M_; i++ ){
			const short * key = hash_table.getKey( i );
			for( int k=0; k<d_; k++ ){
				n1[k] = key[k] - 1;
				n2[k] = key[k] + 1;
			}
			n1[j] = key[j] + d_;
			n2[j] = key[j] - d_;
			
			blur_neighbors_[j*M_+i].n1 = hash_table.find( n1 );
			blur_neighbors_[j*M_+i].n2 = hash_table.find( n2 );
		}
	}
	delete[] n1;
	delete[] n2;
}
#endif
void Permutohedral::seqCompute ( float* out, const float* in, int value_size, bool reverse ) const
{
	// Shift all values by 1 such that -1 -> 0 (used for blurring)
	float * values = new float[ (M_+2)*value_size ];
	float * new_values = new float[ (M_+2)*value_size ];
	
	for( int i=0; i<(M_+2)*value_size; i++ )
		values[i] = new_values[i] = 0;
	
	// Splatting
	for( int i=0;  i<N_; i++ ){
		for( int j=0; j<=d_; j++ ){
			int o = offset_[i*(d_+1)+j]+1;
			float w = barycentric_[i*(d_+1)+j];
			for( int k=0; k<value_size; k++ )
				values[ o*value_size+k ] += w * in[ i*value_size+k ];
		}
	}
	
	for( int j=reverse?d_:0; j<=d_ && j>=0; reverse?j--:j++ ){
		for( int i=0; i<M_; i++ ){
			float * old_val = values + (i+1)*value_size;
			float * new_val = new_values + (i+1)*value_size;
			
			int n1 = blur_neighbors_[j*M_+i].n1+1;
			int n2 = blur_neighbors_[j*M_+i].n2+1;
			float * n1_val = values + n1*value_size;
			float * n2_val = values + n2*value_size;
			for( int k=0; k<value_size; k++ )
				new_val[k] = old_val[k]+0.5*(n1_val[k] + n2_val[k]);
		}
		std::swap( values, new_values );
	}
	// Alpha is a magic scaling constant (write Andrew if you really wanna understand this)
	float alpha = 1.0f / (1+powf(2, -d_));
	
	// Slicing
	for( int i=0; i<N_; i++ ){
		for( int k=0; k<value_size; k++ )
			out[i*value_size+k] = 0;
		for( int j=0; j<=d_; j++ ){
			int o = offset_[i*(d_+1)+j]+1;
			float w = barycentric_[i*(d_+1)+j];
			for( int k=0; k<value_size; k++ )
				out[ i*value_size+k ] += w * values[ o*value_size+k ] * alpha;
		}
	}
	
	
	delete[] values;
	delete[] new_values;
}
#ifdef SSE_PERMUTOHEDRAL
void Permutohedral::sseCompute ( float* out, const float* in, int value_size, bool reverse ) const
{
	const int sse_value_size = (value_size-1)*sizeof(float) / sizeof(__m128) + 1;
	// Shift all values by 1 such that -1 -> 0 (used for blurring)
	__m128 * sse_val    = (__m128*) _mm_malloc( sse_value_size*sizeof(__m128), 16 );
	__m128 * values     = (__m128*) _mm_malloc( (M_+2)*sse_value_size*sizeof(__m128), 16 );
	__m128 * new_values = (__m128*) _mm_malloc( (M_+2)*sse_value_size*sizeof(__m128), 16 );
	
	__m128 Zero = _mm_set1_ps( 0 );
	
	for( int i=0; i<(M_+2)*sse_value_size; i++ )
		values[i] = new_values[i] = Zero;
	for( int i=0; i<sse_value_size; i++ )
		sse_val[i] = Zero;
	
	// Splatting
	for( int i=0;  i<N_; i++ ){
		memcpy( sse_val, in+i*value_size, value_size*sizeof(float) );
		for( int j=0; j<=d_; j++ ){
			int o = offset_[i*(d_+1)+j]+1;
			__m128 w = _mm_set1_ps( barycentric_[i*(d_+1)+j] );
			for( int k=0; k<sse_value_size; k++ )
				values[ o*sse_value_size+k ] += w * sse_val[k];
		}
	}
	// Blurring
	__m128 half = _mm_set1_ps(0.5);
	for( int j=reverse?d_:0; j<=d_ && j>=0; reverse?j--:j++ ){
		for( int i=0; i<M_; i++ ){
			__m128 * old_val = values + (i+1)*sse_value_size;
			__m128 * new_val = new_values + (i+1)*sse_value_size;
			
			int n1 = blur_neighbors_[j*M_+i].n1+1;
			int n2 = blur_neighbors_[j*M_+i].n2+1;
			__m128 * n1_val = values + n1*sse_value_size;
			__m128 * n2_val = values + n2*sse_value_size;
			for( int k=0; k<sse_value_size; k++ )
				new_val[k] = old_val[k]+half*(n1_val[k] + n2_val[k]);
		}
		std::swap( values, new_values );
	}
	// Alpha is a magic scaling constant (write Andrew if you really wanna understand this)
	float alpha = 1.0f / (1+powf(2, -d_));
	
	// Slicing
	for( int i=0; i<N_; i++ ){
		for( int k=0; k<sse_value_size; k++ )
			sse_val[ k ] = Zero;
		for( int j=0; j<=d_; j++ ){
			int o = offset_[i*(d_+1)+j]+1;
			__m128 w = _mm_set1_ps( barycentric_[i*(d_+1)+j] * alpha );
			for( int k=0; k<sse_value_size; k++ )
				sse_val[ k ] += w * values[ o*sse_value_size+k ];
		}
		memcpy( out+i*value_size, sse_val, value_size*sizeof(float) );
	}
	
	_mm_free( sse_val );
	_mm_free( values );
	_mm_free( new_values );
}
#else
void Permutohedral::sseCompute ( float* out, const float* in, int value_size, bool reverse ) const
{
	seqCompute( out, in, value_size, reverse );
}
#endif
void Permutohedral::compute ( MatrixXf & out, const MatrixXf & in, bool reverse ) const
{
	if( out.cols() != in.cols() || out.rows() != in.rows() )
		out = 0*in;
	if( in.rows() <= 2 )
		seqCompute( out.data(), in.data(), in.rows(), reverse );
	else
		sseCompute( out.data(), in.data(), in.rows(), reverse );
}
MatrixXf Permutohedral::compute ( const MatrixXf & in, bool reverse ) const
{
	MatrixXf r;
	compute( r, in, reverse );
	return r;
}
// Compute the gradient of a^T K b
void Permutohedral::gradient ( float* df, const float * a, const float* b, int value_size ) const
{
	// Shift all values by 1 such that -1 -> 0 (used for blurring)
	float * values = new float[ (M_+2)*value_size ];
	float * new_values = new float[ (M_+2)*value_size ];
	
	// Set the results to 0
	std::fill( df, df+N_*d_, 0.f );
	
	// Initialize some constants
	std::vector<float> scale_factor( d_ );
	float inv_std_dev = sqrt(2.0 / 3.0)*(d_+1);
	for( int i=0; i<d_; i++ )
		scale_factor[i] = 1.0 / sqrt( double((i+2)*(i+1)) ) * inv_std_dev;
	
	// Alpha is a magic scaling constant multiplied by down_factor
	float alpha = 1.0f / (1+powf(2, -d_)) / (d_+1);
	
	for( int dir=0; dir<2; dir++ ) {
		for( int i=0; i<(M_+2)*value_size; i++ )
			values[i] = new_values[i] = 0;
	
		// Splatting
		for( int i=0;  i<N_; i++ ){
			for( int j=0; j<=d_; j++ ){
				int o = offset_[i*(d_+1)+j]+1;
				float w = barycentric_[i*(d_+1)+j];
				for( int k=0; k<value_size; k++ )
					values[ o*value_size+k ] += w * (dir?b:a)[ i*value_size+k ];
			}
		}
		
		// BLUR
		for( int j=dir?d_:0; j<=d_ && j>=0; dir?j--:j++ ){
			for( int i=0; i<M_; i++ ){
				float * old_val = values + (i+1)*value_size;
				float * new_val = new_values + (i+1)*value_size;
			
				int n1 = blur_neighbors_[j*M_+i].n1+1;
				int n2 = blur_neighbors_[j*M_+i].n2+1;
				float * n1_val = values + n1*value_size;
				float * n2_val = values + n2*value_size;
				for( int k=0; k<value_size; k++ )
					new_val[k] = old_val[k]+0.5*(n1_val[k] + n2_val[k]);
			}
			std::swap( values, new_values );
		}
	
		// Slicing gradient computation
		std::vector<float> r_a( (d_+1)*value_size ), sm( value_size );
	
		for( int i=0; i<N_; i++ ){
			// Rotate a
			std::fill( r_a.begin(), r_a.end(), 0.f );
			for( int j=0; j<=d_; j++ ){
				int r0 = d_ - rank_[i*(d_+1)+j];
				int r1 = r0+1>d_?0:r0+1;
				int o0 = offset_[i*(d_+1)+r0]+1;
				int o1 = offset_[i*(d_+1)+r1]+1;
				for( int k=0; k<value_size; k++ ) {
					r_a[ j*value_size+k ] += alpha*values[ o0*value_size+k ];
					r_a[ j*value_size+k ] -= alpha*values[ o1*value_size+k ];
				}
			}
			// Multiply by the elevation matrix
			std::copy( r_a.begin(), r_a.begin()+value_size, sm.begin() );
			for( int j=1; j<=d_; j++ ) {
				float grad = 0;
				for( int k=0; k<value_size; k++ ) {
					// Elevate ...
					float v = scale_factor[j-1]*(sm[k]-j*r_a[j*value_size+k]);
					// ... and add
					grad += (dir?a:b)[ i*value_size+k ]*v;
				
					sm[k] += r_a[j*value_size+k];
				}
				// Store the gradient
				df[i*d_+j-1] += grad;
			}
		}
	}		
	delete[] values;
	delete[] new_values;
}
