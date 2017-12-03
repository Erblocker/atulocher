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

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "psyc.h"
#include "utils.h"

static unsigned char randomSeeded = 0;

void PSErr(const char* tag, char* fmt, ...) {
    va_list args;
    
    fflush (stdout);
    if (PSGlobalFlags & FLAG_LOG_COLORS) fprintf(stderr, RED);
    fprintf(stderr, "ERROR");
    if (tag != NULL) fprintf(stderr, " [%s]: ", tag);
    else fprintf(stderr, ": ");
    
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    
    fprintf(stderr, "\n");
    if (PSGlobalFlags & FLAG_LOG_COLORS) fprintf(stderr, WHITE);
}

/* Activation Functions */

double sigmoid(double val) {
    return 1.0 / (1.0 + exp(-val));
}

double sigmoid_derivative(double val) {
    return val * (1 - val);
}

double relu(double val) {
    return (val >= 0.0 ? val : 0.0);
}

double relu_derivative(double val) {
    return (double)(val > 0.0);
}

double tanh_derivative(double val) {
    return (1 - (val * val));
}

/* Network Functions */

void PSAbortLayer(PSNeuralNetwork * network, PSLayer * layer) {
    if (!network->size) return;
    if (layer->index == (network->size - 1)) {
        network->size--;
        if (!network->size) {
            network->input_size = 0;
            network->output_size = 0;
        } else {
            PSLayer * outputLayer = network->layers[network->size - 1];
            if (outputLayer) network->output_size = outputLayer->size;
            else network->output_size = 0;
            PSLayer * inputLayer = network->layers[0];
            if (inputLayer) network->input_size = inputLayer->size;
            else network->input_size = 0;
        }
        PSDeleteLayer(layer);
    }
}

/* Misc */


double normalized_random() {
    if (!randomSeeded) {
        randomSeeded = 1;
        srand(time(NULL));
    }
    int r = rand();
    return ((double) r / (double) RAND_MAX);
}

double gaussian_random(double mean, double stddev) {
    double theta = 2 * M_PI * normalized_random();
    double rho = sqrt(-2 * log(1 - normalized_random()));
    double scale = stddev * rho;
    double x = mean + scale * cos(theta);
    double y = mean + scale * sin(theta);
    double r = normalized_random();
    return (r > 0.5 ? y : x);
}
