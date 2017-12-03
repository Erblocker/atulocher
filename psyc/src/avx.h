/*
 Copyright (c) 2016 Fabio Nicotra.
 All rights reserved.
 
 Redistribution and use in source and binary forms are permitted
 provided that the above copyright notice and this paragraph are
 duplicated in all such forms and that any documentation,
 advertising materials, and other materials related to such
 distribution and use acknowledge that the software was developed
 by the copyright holder. The name of the
 copyright holder may not be used to endorse or promote products derived
 from this software without specific prior written permission.
 THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
 IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef __PS_AVX_H
#define __PS_AVX_H

#define AVXGetStepLen(s) (s >= AVX_VECTOR_SIZE ? AVX_VECTOR_SIZE : \
    AVX_VECTOR_SIZE / 2)
#define AVXGetDotStepLen(s) (s >= AVX_VECTOR4_SIZE ? AVX_VECTOR4_SIZE : \
    (s >= AVX_VECTOR2_SIZE ? AVX_VECTOR2_SIZE : \
    (s >= AVX_VECTOR_SIZE ? AVX_VECTOR_SIZE : AVX_VECTOR_SIZE/2)))
#define AVXGetDotProductFunc(s) (s >= AVX_VECTOR4_SIZE ? avx_dot_product16 : \
    (s >= AVX_VECTOR2_SIZE ? avx_dot_product8 : \
    (s >= AVX_VECTOR_SIZE ? avx_dot_product4 : avx_dot_product2)))
#define AVXGetMultiplyValFunc(s) (s >= AVX_VECTOR_SIZE ? avx_multiply_value4 : \
    avx_multiply_value2)
#define AVXGetMultiplyFunc(s) (s >= AVX_VECTOR_SIZE ? avx_multiply4 : \
    avx_multiply2)
#define AVXGetSumFunc(s) (s >= AVX_VECTOR_SIZE ? avx_sum4 : \
    avx_sum2)
#define AVXGetDiffFunc(s) (s >= AVX_VECTOR_SIZE ? avx_diff4 : \
    avx_diff2)

#define AVX_STORE_MODE_NORM 0
#define AVX_STORE_MODE_ADD  1
#define AVX_STORE_MODE_SUB  2

extern int AVX_VECTOR_SIZE;
extern int AVX_VECTOR4_SIZE;
extern int AVX_VECTOR2_SIZE;

typedef double (* avx_dot_product)(double * x, double * y);
typedef void (* avx_multiply_value)(double * x, double v, double * d, int mode);
typedef void (* avx_multiply)(double * x, double * y, double * dest, int mode);
typedef void (* avx_sum)(double * x, double * y, double * dest, int mode);
typedef void (* avx_diff)(double * x, double * y, double * dest, int mode);

double avx_dot_product2(double * x, double * y);
double avx_dot_product4(double * x, double * y);
double avx_dot_product8(double * x, double * y);
double avx_dot_product16(double * x, double * y);

void avx_multiply_value2(double * x, double value, double * dest, int mode);
void avx_multiply2(double * x, double * y, double * dest, int mode);
void avx_multiply_value4(double * x, double value, double * dest, int mode);
void avx_multiply4(double * x, double * y, double * dest, int mode);

void avx_sum2(double * x, double * y, double * dest, int mode);
void avx_sum4(double * x, double * y, double * dest, int mode);
void avx_diff2(double * x, double * y, double * dest, int mode);
void avx_diff4(double * x, double * y, double * dest, int mode);

#endif //__PS_AVX_H
