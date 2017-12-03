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

#ifndef __PSYC_H
#define __PSYC_H

#define PSYC_VERSION      "0.2.2"

#define LAYER_TYPES  6

#define STATUS_UNTRAINED    0
#define STATUS_TRAINED      1
#define STATUS_TRAINING     2
#define STATUS_ERROR        3

#define NULL_VALUE -9999999.99

#define FLAG_NONE 0
#define FLAG_RECURRENT  (1 << 0)
#define FLAG_ONEHOT     (1 << 1)

/* Global Flags*/

#define FLAG_LOG_COLORS (1 << 0)

#define TRAINING_NO_SHUFFLE     (1 << 0)
#define TRAINING_ADJUST_RATE    (1 << 1)

#define BPTT_TRUNCATE   4


typedef double  (*PSActivationFunction) (double);
typedef int     (*PSFeedforwardFunction) (void * network, void * layer, ...);
typedef double  (*PSLossFunction) (double* x, double* y, int size,
                                   int onehot_size);
typedef void    (*PSTrainCallback) (void * network, int epoch, double loss,
                                    double previous_loss, float accuracy,
                                    double * rate);

typedef struct {
    double bias;
    double * weights;
} PSGradient;

typedef enum {
    FullyConnected,
    Convolutional,
    Pooling,
    Recurrent,
    LSTM,
    SoftMax
} PSLayerType;

typedef struct {
    int count;
    double * parameters;
} PSLayerParameters;

typedef struct {
    int feature_count;
    int weights_size;
    double * biases;
    double ** weights;
} PSSharedParams;

typedef struct {
    int flags;
    double l2_decay;
} PSTrainingOptions;

typedef struct {
    int index;
    int weights_size;
    double bias;
    double * weights;
    double activation;
    double z_value;
    void * extra;
    void * layer;
} PSNeuron;

typedef struct {
    PSLayerType type;
    int index;
    int size;
    PSLayerParameters * parameters;
    PSActivationFunction activate;
    PSActivationFunction derivative;
    PSFeedforwardFunction feedforward;
    PSNeuron ** neurons;
    double * delta;
    int flags;
    void * extra;
#ifdef USE_AVX
    double * avx_activation_cache;
#endif
    void * network;
} PSLayer;

typedef struct {
    const char * name;
    int size;
    PSLayer ** layers;
    PSLossFunction loss;
    int flags;
    unsigned char status;
    int input_size;
    int output_size;
    int current_epoch;
    int current_batch;
    PSTrainCallback onEpochTrained;
} PSNeuralNetwork;

extern int PSGlobalFlags;

PSNeuralNetwork * PSCreateNetwork(const char* name);
PSNeuralNetwork * PSCloneNetwork(PSNeuralNetwork * network, int layout_only);
int PSLoadNetwork(PSNeuralNetwork * network, const char* filename);
int PSSaveNetwork(PSNeuralNetwork * network, const char* filename);
PSLayer * PSAddLayer(PSNeuralNetwork * network, PSLayerType type, int size,
                     PSLayerParameters* params);
PSLayer * PSAddConvolutionalLayer(PSNeuralNetwork * network,
                                  PSLayerParameters* params);
PSLayer * PSAddPoolingLayer(PSNeuralNetwork * network,
                            PSLayerParameters* params);
PSLayerParameters * PSCreateLayerParamenters(int count, ...);
int PSSetLayerParameter(PSLayerParameters * params, int param, double value);
int PSAddLayerParameter(PSLayerParameters * params, double val);
PSLayerParameters * PSCreateConvolutionalParameters(double feature_count,
                                                    double region_size,
                                                    int stride,
                                                    int padding,
                                                    int use_relu);
void PSDeleteLayerParamenters(PSLayerParameters * params);
int PSFeedforward(PSNeuralNetwork * network, double * values);
int PSClassify(PSNeuralNetwork * network, double * values);

void PSDeleteNetwork(PSNeuralNetwork * network);
void PSDeleteLayer(PSLayer * layer);
void PSDeleteNeuron(PSNeuron * neuron, PSLayer * layer);
void PSDeleteGradients(PSGradient ** gradients, PSNeuralNetwork * network);

void PSTrain(PSNeuralNetwork * network,
             double * training_data,
             int data_size,
             int epochs,
             double learning_rate,
             int batch_size,
             PSTrainingOptions * options,
             double * test_data,
             int test_size);
float PSTest(PSNeuralNetwork * network, double * test_data, int data_size);
int PSVerifyNetwork(PSNeuralNetwork * network);
//int arrayMaxIndex(double * array, int len);
char * PSGetLabelForType(PSLayerType type);
char * PSGetLayerTypeLabel(PSLayer * layer);
void PSPrintNetworkInfo(PSNeuralNetwork * network);

// Loss functions

double PSQuadraticLoss(double * x, double * y, int size, int onehot_size);
double PSCrossEntropyLoss(double * x, double * y, int size, int onehot_size);

#endif // __PSYC_H


