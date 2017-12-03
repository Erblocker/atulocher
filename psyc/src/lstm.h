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

#ifndef __PS_LSTM_H
#define __PS_LSTM_H

#include "psyc.h"

#define GetLSTMCell(neuron) ((PSLSTMCell*) neuron->extra)
#define GetLSTMGradientBiases(n, gradient) (gradient->weights + n->weights_size)

typedef struct {
    int states_count;
    int weights_size;
    double * states;
    double * z_values;
    double * candidates;
    double * input_gates;
    double * output_gates;
    double * forget_gates;
    double candidate_bias;
    double input_bias;
    double output_bias;
    double forget_bias;
    double * candidate_weights;
    double * input_weights;
    double * output_weights;
    double * forget_weights;
} PSLSTMCell;

PSLSTMCell * PSCreateLSTMCell(PSNeuron * neuron, int lsize);
void PSDeleteLSTMCell(PSLSTMCell * cell);
void PSUpdateLSTMBiases(PSNeuron * neuron, PSGradient * gradient, double rate);

/* Init Functions */

int PSInitLSTMLayer(PSNeuralNetwork * network, PSLayer * layer,
                    int size, int ws);

/* Feedforward Functions */

int PSLSTMFeedforward(void * _net, void * _layer, ...);

/* Backpropagation Functions */

int PSLSTMBackprop(PSLayer * layer, PSLayer * previousLayer,
                   PSGradient * lgradients, int t);


#endif // __PS_LSTM_H
