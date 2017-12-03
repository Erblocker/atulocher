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

#ifndef __PS_UTILS_H
#define __PS_UTILS_H

#include <math.h>

#ifndef M_PI
#define M_PI 3.141592653589793
#endif

#define getNeuronLayer(neuron) ((PSLayer*) neuron->layer)
#define getLayerNetwork(layer) ((PSNeuralNetwork*) layer->network)
#define shouldApplyDerivative(network) (network->loss != PSCrossEntropyLoss)

#ifdef USE_AVX

#define AVXDotProduct(size, x, y, res, i, is_recurrent, t) do { \
    int avx_step_len = AVXGetDotStepLen(size); \
    avx_dot_product dot_product = AVXGetDotProductFunc(size); \
    int avx_steps = size / avx_step_len, avx_step; \
    for (avx_step = 0; avx_step < avx_steps; avx_step++) { \
        double * x_vector = x + i; \
        if (is_recurrent) x_vector += (t * size); \
        double * y_vector = y + i; \
        res += dot_product(x_vector, y_vector); \
        i += avx_step_len; \
    } \
} while (0)


#define AVXDotSquare(size, x, res, i, is_recurrent, t) do {\
    int avx_step_len = AVXGetDotStepLen(size); \
    avx_dot_product dot_product = AVXGetDotProductFunc(size); \
    int avx_steps = size / avx_step_len, avx_step; \
    for (avx_step = 0; avx_step < avx_steps; avx_step++) { \
        double * x_vector = x + i; \
        if (is_recurrent) x_vector += (t * size); \
        res += dot_product(x_vector, x_vector); \
        i += avx_step_len; \
    } \
} while (0)

#define AVXMultiplyValue(size, x, val, dest, i, is_recurrent, t, mode) do { \
    int avx_step_len = AVXGetStepLen(size); \
    int avx_steps = size / avx_step_len, avx_step; \
    avx_multiply_value multiply_val = AVXGetMultiplyValFunc(size); \
    for (avx_step = 0; avx_step < avx_steps; avx_step++) { \
        double * x_vector = x + i; \
        if (is_recurrent) x_vector += (t * size); \
        multiply_val(x_vector, val, dest + i, mode); \
        i += avx_step_len; \
    } \
} while (0)

#define AVXMultiplyValues(size, x1, v1, x2, v2, d, i, is_rec, t, m1, m2) do {\
    int avx_step_len = AVXGetStepLen(size); \
    int avx_steps = size / avx_step_len, avx_step; \
    avx_multiply_value multiply_val = AVXGetMultiplyValFunc(size); \
    for (avx_step = 0; avx_step < avx_steps; avx_step++) { \
        double * xv1 = x1 + i; \
        double * xv2 = x2 + i; \
        double * dd = d + i; \
        if (is_rec) {\
            xv1 += (t * size); \
            xv2 += (t * size); \
        }\
        multiply_val(xv1, v1, dd, m1); \
        multiply_val(xv2, v2, dd, m2); \
        i += avx_step_len; \
    } \
} while (0)

#define AVXSum(size, x, y, dest, i, mode) do { \
    int avx_step_len = AVXGetStepLen(size); \
    int avx_steps = size / avx_step_len, avx_step; \
    int x_is_dest = (x == dest); \
    avx_sum __avx_sum = AVXGetSumFunc(size); \
    for (avx_step = 0; avx_step < avx_steps; avx_step++) { \
        int doffs = (x_is_dest ? i : 0); \
        __avx_sum(x + i, y + i, dest + doffs, mode); \
        i += avx_step_len; \
    } \
} while (0)

#define AVXDiff(size, x, y, dest, i, mode) do { \
    int avx_step_len = AVXGetStepLen(size); \
    int avx_steps = size / avx_step_len, avx_step; \
    int x_is_dest = (x == dest); \
    avx_sum __avx_diff = AVXGetDiffFunc(size); \
    for (avx_step = 0; avx_step < avx_steps; avx_step++) { \
        int doffs = (x_is_dest ? i : 0); \
        __avx_diff(x + i, y + i, dest + doffs, mode); \
        i += avx_step_len; \
    } \
} while (0)

#endif

#define RED     "\x1b[31m"
#define GREEN   "\x1b[32m"
#define YELLOW  "\x1b[33m"
#define BLUE    "\x1b[34m"
#define MAGENTA "\x1b[35m"
#define CYAN    "\x1b[36m"
#define WHITE   "\x1b[97m"
#define BOLD    "\x1b[1m"
#define DIM     "\x1b[2m"
#define HIDDEN  "\x1b[8m"
#define RESET   "\x1b[0m"
#define RESET_BOLD "\x1b[21m"

#define printMemoryErrorMsg() PSErr(NULL, "Could not allocate memory!")

void PSErr(const char* tag, char* fmt, ...);

/* Activation Functions */

double sigmoid(double val);

double sigmoid_derivative(double val);

double relu(double val);

double relu_derivative(double val);

double tanh_derivative(double val);

/* Network Functions */

void PSAbortLayer(PSNeuralNetwork * network, PSLayer * layer);

/* Misc */


double normalized_random();

double gaussian_random(double mean, double stddev);

#endif //__PS_UTILS_H
