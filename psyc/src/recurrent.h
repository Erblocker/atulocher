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

#ifndef __PS_RECURRENT_H
#define __PS_RECURRENT_H

#include "psyc.h"

#define GetRecurrentCell(neuron) ((PSRecurrentCell*) neuron->extra)

typedef struct {
    int states_count;
    int weights_size;
    double * states;
    double * weights;
} PSRecurrentCell;

PSRecurrentCell * PSCreateRecurrentCell(PSNeuron * neuron, int lsize);
double * PSAddRecurrentState(PSNeuron * neuron, double state, int times, int t);

/* Init Functions */

int PSInitRecurrentLayer(PSNeuralNetwork * network, PSLayer * layer,
                         int size, int ws);

/* Feedforward Functions */

int PSRecurrentFeedforward(void * _net, void * _layer, ...);

/* Backpropagation Functions */

int PSRecurrentBackprop(PSLayer * layer, PSLayer * previousLayer, int lowest_t,
                        PSGradient * lgradients, int t);

#endif //__PS_RECURRENT_H
