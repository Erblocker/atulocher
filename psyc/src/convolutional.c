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
#include <math.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <assert.h>

#ifdef USE_AVX
#include "avx.h"
#endif

#include "psyc.h"
#include "utils.h"
#include "convolutional.h"
#include "recurrent.h"

double getDeltaForConvolutionalNeuron(PSNeuron * neuron,
                                      PSLayer * layer,
                                      PSLayer * nextLayer,
                                      double * last_delta)
{
    
    int index = neuron->index, i, j, row, col;
    int size = layer->size;
    PSLayerParameters * lparams = layer->parameters;
    int feature_count = (int) (lparams->parameters[PARAM_FEATURE_COUNT]);
    double output_w = lparams->parameters[PARAM_OUTPUT_WIDTH];
    int feature_size = size / feature_count;
    int feature_idx = index / feature_size;
    PSLayerParameters * nparams = nextLayer->parameters;
    int next_feature_count = (int) (nparams->parameters[PARAM_FEATURE_COUNT]);
    int next_region_size = (int) (nparams->parameters[PARAM_REGION_SIZE]);
    int stride = (int) (nparams->parameters[PARAM_STRIDE]);
    double next_output_w = nparams->parameters[PARAM_OUTPUT_WIDTH];
    int next_feature_size = nextLayer->size / next_feature_count;
    int features_step = next_feature_count / feature_count;
    int next_feature_idx = feature_idx * features_step;
    int max_feature_idx = next_feature_idx + features_step;
    
    int n_col = index % (int) output_w;
    int n_row = index / (int) output_w;
    
    PSSharedParams * shared = getConvSharedParams(nextLayer);
    if (shared == NULL) {
        //TODO: handle shared == NULL
        return 0;
    }
    
    double dv = 0;
    
    for (i = next_feature_idx; i < max_feature_idx; i++) {
        double * weights = shared->weights[i];
        int offset = i * next_feature_size;
        row = 0;
        col = 0;
        for (j = 0; j < next_feature_size; j++) {
            int idx = offset + j;
            double d = last_delta[idx];
            col = idx % (int) next_output_w;
            if (col == 0 && j > 0) row++;
            int r_row = row * stride;
            int r_col = col * stride;
            if (r_col > n_col || r_row > n_row) break;
            int max_x = next_region_size + r_col;
            int max_y = next_region_size + r_row;
            if ((n_col >= r_col && n_col < max_x) &&
                (n_row >= r_row && n_row < max_y)) {
                int widx = (r_row * next_region_size) + r_col;
                dv += (d * weights[widx]);
            }
        }
    }
    
    if (layer->derivative != NULL)
        dv *= layer->derivative(neuron->activation);
    return dv;
}

/* Init Functions */


int PSInitConvolutionalLayer(PSNeuralNetwork * network, PSLayer * layer,
                             PSLayerParameters * parameters) {
    int index = layer->index;
    char * func = "initConvolutionalLayer";
    PSLayer * previous = network->layers[index - 1];
    if (parameters == NULL) {
        PSErr(func, "Layer parameters is NULL!");
        PSAbortLayer(network, layer);
        return 0;
    }
    if (parameters->count < CONV_PARAMETER_COUNT) {
        PSErr(func, "Convolutional Layer parameters count must be %d",
              CONV_PARAMETER_COUNT);
        PSAbortLayer(network, layer);
        return 0;
    }
    double * params = parameters->parameters;
    int feature_count = (int) (params[PARAM_FEATURE_COUNT]);
    if (feature_count <= 0) {
        PSErr(func, "FEATURE_COUNT must be > 0 (given: %d)", feature_count);
        PSAbortLayer(network, layer);
        return 0;
    }
    double region_size = params[PARAM_REGION_SIZE];
    if (region_size <= 0) {
        PSErr(func, "REGION_SIZE must be > 0 (given: %lf)", region_size);
        PSAbortLayer(network, layer);
        return 0;
    }
    int previous_size = previous->size;
    PSLayerParameters * previous_params = previous->parameters;
    double input_w, input_h, output_w, output_h;
    int use_relu = (int) (params[PARAM_USE_RELU]);
    if (previous_params == NULL) {
        double w = sqrt(previous_size);
        input_w = w; input_h = w;
        previous_params = PSCreateConvolutionalParameters(1, 0, 0, 0, 0);
        previous_params->parameters[PARAM_OUTPUT_WIDTH] = input_w;
        previous_params->parameters[PARAM_OUTPUT_HEIGHT] = input_h;
        previous->parameters = previous_params;
    } else {
        input_w = previous_params->parameters[PARAM_OUTPUT_WIDTH];
        input_h = previous_params->parameters[PARAM_OUTPUT_HEIGHT];
        int prev_features = 1;
        if (previous->type == Pooling) {
            prev_features =
            (int) previous_params->parameters[PARAM_FEATURE_COUNT];
            int is_valid = 1;
            if (feature_count < prev_features) {
                PSErr(func,"FEATURE_COUNT %d cannot be < than previous %d one",
                      feature_count, prev_features);
                is_valid = 0;
            } else if ((feature_count % prev_features) != 0) {
                PSErr(func,"FEATURE_COUNT %d must be a multiple than "
                      "previous %d one", feature_count, prev_features);
                is_valid = 0;
            }
            if (!is_valid) {
                PSAbortLayer(network, layer);
                return 0;
            }
        }
        double prev_area = input_w * input_h * (double) prev_features;
        if ((int) prev_area != previous_size) {
            PSErr(func, "Previous size %d != %lfx%lf",
                  previous_size, input_w, input_h);
            PSAbortLayer(network, layer);
            return 0;
        }
    }
    params[PARAM_INPUT_WIDTH] = input_w;
    params[PARAM_INPUT_HEIGHT] = input_h;
    int stride = (int) params[PARAM_STRIDE];
    int padding = (int) params[PARAM_PADDING];
    if (stride == 0) stride = 1;
    output_w =  calculateConvolutionalSide(input_w, region_size,
                                           (double) stride, (double) padding);
    output_h =  calculateConvolutionalSide(input_h, region_size,
                                           (double) stride, (double) padding);
    params[PARAM_OUTPUT_WIDTH] = output_w;
    params[PARAM_OUTPUT_HEIGHT] = output_h;
    int area = (int)(output_w * output_h);
    int size = area * feature_count;
    layer->size = size;
    layer->neurons = malloc(sizeof(PSNeuron*) * size);
    if (layer->neurons == NULL) {
        PSErr(func, "Layer[%d]: Could not allocate neurons!", index);
        PSAbortLayer(network, layer);
        return 0;
    }
#ifdef USE_AVX
    layer->avx_activation_cache = calloc(size, sizeof(double));
    if (layer->avx_activation_cache == NULL) {
        printMemoryErrorMsg();
        PSAbortLayer(network, layer);
        return 0;
    }
#endif
    PSSharedParams * shared = malloc(sizeof(PSSharedParams));
    if (shared == NULL) {
        PSErr(func, "Layer[%d]: Couldn't allocate shared params!", index);
        PSAbortLayer(network, layer);
        return 0;
    }
    shared->feature_count = feature_count;
    shared->weights_size = (int)(region_size * region_size);
    shared->biases = malloc(feature_count * sizeof(double));
    shared->weights = malloc(feature_count * sizeof(double*));
    if (shared->biases == NULL || shared->weights == NULL) {
        PSErr(func, "Layer[%d]: Could not allocate memory!", index);
        PSAbortLayer(network, layer);
        return 0;
    }
    layer->extra = shared;
    int i, j, w;
    for (i = 0; i < feature_count; i++) {
        shared->biases[i] = gaussian_random(0, 1);
        shared->weights[i] = malloc(shared->weights_size * sizeof(double));
        if (shared->weights[i] == NULL) {
            PSErr(func, "Layer[%d]: Could not allocate weights!", index);
            PSAbortLayer(network, layer);
            return 0;
        }
        for (w = 0; w < shared->weights_size; w++) {
            shared->weights[i][w] = gaussian_random(0, 1);
        }
        for (j = 0; j < area; j++) {
            int idx = (i * area) + j;
            PSNeuron * neuron = malloc(sizeof(PSNeuron));
            if (neuron == NULL) {
                PSErr(func, "Layer[%d]: Couldn't allocate neuron!",index);
                PSAbortLayer(network, layer);
                return 0;
            }
            neuron->index = idx;
            neuron->extra = NULL;
            neuron->weights_size = shared->weights_size;
            neuron->bias = shared->biases[i];
            neuron->weights = shared->weights[i];
            neuron->layer = layer;
            layer->neurons[idx] = neuron;
        }
    }
    if (!use_relu) {
        layer->activate = sigmoid;
        layer->derivative = sigmoid_derivative;
    } else {
        layer->activate = relu;
        layer->derivative = relu_derivative;
    }
    layer->feedforward = PSConvolve;
    return 1;
}

int PSInitPoolingLayer(PSNeuralNetwork * network, PSLayer * layer,
                       PSLayerParameters * parameters) {
    int index = layer->index;
    char * func = "initPoolingLayer";
    PSLayer * previous = network->layers[index - 1];
    if (previous->type != Convolutional) {
        PSErr(func, "Pooling's previous layer must be a Convolutional layer!");
        PSAbortLayer(network, layer);
        return 0;
    }
    if (parameters == NULL) {
        PSErr(func, "Layer parameters is NULL!");
        PSAbortLayer(network, layer);
        return 0;
    }
    if (parameters->count < CONV_PARAMETER_COUNT) {
        PSErr(func, "Convolutional Layer parameters count must be %d",
              CONV_PARAMETER_COUNT);
        PSAbortLayer(network, layer);
        return 0;
    }
    double * params = parameters->parameters;
    PSLayerParameters * previous_parameters = previous->parameters;
    if (previous_parameters == NULL) {
        PSErr(func, "Previous layer parameters is NULL!");
        PSAbortLayer(network, layer);
        return 0;
    }
    if (previous_parameters->count < CONV_PARAMETER_COUNT) {
        PSErr(func, "Convolutional Layer parameters count must be %d",
              CONV_PARAMETER_COUNT);
        PSAbortLayer(network, layer);
        return 0;
    }
    double * previous_params = previous_parameters->parameters;
    int feature_count = (int) (previous_params[PARAM_FEATURE_COUNT]);
    params[PARAM_FEATURE_COUNT] = (double) feature_count;
    double region_size = params[PARAM_REGION_SIZE];
    if (region_size <= 0) {
        PSErr(func, "REGION_SIZE must be > 0 (given: %lf)", region_size);
        PSAbortLayer(network, layer);
        return 0;
    }
    double input_w, input_h, output_w, output_h;
    input_w = previous_params[PARAM_OUTPUT_WIDTH];
    input_h = previous_params[PARAM_OUTPUT_HEIGHT];
    params[PARAM_INPUT_WIDTH] = input_w;
    params[PARAM_INPUT_HEIGHT] = input_h;
    
    output_w = calculatePoolingSide(input_w, region_size);
    output_h = calculatePoolingSide(input_h, region_size);
    params[PARAM_OUTPUT_WIDTH] = output_w;
    params[PARAM_OUTPUT_HEIGHT] = output_h;
    int area = (int)(output_w * output_h);
    int size = area * feature_count;
    layer->size = size;
    layer->neurons = malloc(sizeof(PSNeuron*) * size);
    if (layer->neurons == NULL) {
        PSErr(func, "Layer[%d]: Could not allocate neurons!", index);
        PSAbortLayer(network, layer);
        return 0;
    }
#ifdef USE_AVX
    layer->avx_activation_cache = calloc(size, sizeof(double));
    if (layer->avx_activation_cache == NULL) {
        printMemoryErrorMsg();
        PSAbortLayer(network, layer);
        return 0;
    }
#endif
    int i, j;
    for (i = 0; i < feature_count; i++) {
        for (j = 0; j < area; j++) {
            int idx = (i * area) + j;
            PSNeuron * neuron = malloc(sizeof(PSNeuron));
            if (neuron == NULL) {
                PSErr(func, "Layer[%d]: Couldn't allocate neuron!", index);
                PSAbortLayer(network, layer);
                return 0;
            }
            neuron->index = idx;
            neuron->extra = NULL;
            neuron->weights_size = 0;
            neuron->bias = NULL_VALUE;
            neuron->weights = NULL;
            neuron->layer = layer;
            layer->neurons[idx] = neuron;
        }
    }
    layer->activate = NULL;
    layer->derivative = previous->derivative;
    layer->feedforward = PSPool;
    return 1;
}

/* Feedforward Functions */

int PSConvolve(void * _net, void * _layer, ...) {
    PSNeuralNetwork * net = (PSNeuralNetwork*) _net;
    PSLayer * layer = (PSLayer*) _layer;
    int size = layer->size;
    if (layer->neurons == NULL) {
        PSErr(NULL, "Layer[%d] has no neurons!", layer->index);
        return 0;
    }
    if (layer->index == 0) {
        PSErr(NULL, "Cannot feedforward on layer 0!");
        return 0;
    }
    PSLayer * previous = net->layers[layer->index - 1];
    if (previous == NULL) {
        PSErr(NULL, "Layer[%d]: previous layer is NULL!", layer->index);
        return 0;
    }
    int i, j, x, y, row, col;
    PSLayerParameters * parameters = layer->parameters;
    if (parameters == NULL) {
        PSErr(NULL, "Layer[%d]: parameters are NULL!", layer->index);
        return 0;
    }
    PSLayerParameters * previous_parameters = previous->parameters;
    if (previous_parameters == NULL) {
        PSErr(NULL, "Layer[%d]: parameters are invalid!", layer->index);
        return 0;
    }
    int is_recurrent = (net->flags & FLAG_RECURRENT), times, t;
    if (is_recurrent) {
        va_list args;
        va_start(args, _layer);
        times = va_arg(args, int);
        t = va_arg(args, int);
        va_end(args);
    }
    double * params = parameters->parameters;
    double * previous_params = previous_parameters->parameters;
    int feature_count = (int) (params[PARAM_FEATURE_COUNT]);
    int stride = (int) (params[PARAM_STRIDE]);
    double region_size = params[PARAM_REGION_SIZE];
    double input_w = previous_params[PARAM_OUTPUT_WIDTH];
    double output_w = params[PARAM_OUTPUT_WIDTH];
    int feature_size = size / feature_count;
    PSSharedParams * shared = getConvSharedParams(layer);
    if (shared == NULL) {
        PSErr(NULL, "Layer[%d]: shared params are NULL!", layer->index);
        return 0;
    }
    int previous_feature_size = 0, prev_features = 1, prev_features_step = 1;
    if (previous->type == Pooling) {
        prev_features = (int) (previous_params[PARAM_FEATURE_COUNT]);
        previous_feature_size = previous->size / prev_features;
        prev_features_step = feature_count / prev_features;
    }
    for (i = 0; i < feature_count; i++) {
        double bias = shared->biases[i];
        double * weights = shared->weights[i];
        int previous_feature = 0, feature_offset = 0;
        if (prev_features > 1) {
            previous_feature = i / prev_features_step;
            feature_offset = previous_feature * previous_feature_size;
        }
        row = 0;
        col = 0;
        for (j = 0; j < feature_size; j++) {
            int idx = (i * feature_size) + j;
            PSNeuron * neuron = layer->neurons[idx];
            col = idx % (int) output_w;
            if (col == 0 && j > 0) row++;
            int r_row = row * stride;
            int r_col = col * stride;
            int max_x = region_size + r_col;
            int max_y = region_size + r_row;
            double sum = 0;
            int widx = 0;
            //printf("Neuron %d,%d: r: %d, b: %d\n", col, row, max_x, max_y);
            for (y = r_row; y < max_y; y++) {
                x = r_col;
#ifdef USE_AVX
                int avx_step_len = AVXGetDotStepLen(region_size);
                avx_dot_product dot_product = AVXGetDotProductFunc(region_size);
                int avx_steps = region_size / avx_step_len, avx_step;
                for (avx_step = 0; avx_step < avx_steps; avx_step++) {
                    int nidx = feature_offset + (y * input_w) + x;
                    double * x_vector = previous->avx_activation_cache + nidx;
                    if (is_recurrent) x_vector += (t * previous->size);
                    double * y_vector = weights + widx;
                    sum += dot_product(x_vector, y_vector);
                    x += avx_step_len;
                    widx += avx_step_len;
                }
#endif
                for (; x < max_x; x++) {
                    int nidx = feature_offset + (y * input_w) + x;
                    //printf("  -> %d,%d [%d]\n", x, y, nidx);
                    PSNeuron * prev_neuron = previous->neurons[nidx];
                    double a = prev_neuron->activation;
                    sum += (a * weights[widx++]);
                }
            }
            neuron->z_value = sum + bias;
            neuron->activation = layer->activate(neuron->z_value);
#ifdef USE_AVX
            if (!is_recurrent)
                layer->avx_activation_cache[idx] = neuron->activation;
#endif
            if (is_recurrent) {
                PSAddRecurrentState(neuron, neuron->activation, times, t);
                if (neuron->extra == NULL) {
                    PSErr("convolve", "Failed to allocate Recurrent Cell!");
                    return 0;
                }
            }
        }
    }
    return 1;
}

int PSPool(void * _net, void * _layer, ...) {
    PSNeuralNetwork * net = (PSNeuralNetwork*) _net;
    PSLayer * layer = (PSLayer*) _layer;
    int size = layer->size;
    if (layer->neurons == NULL) {
        PSErr(NULL, "Layer[%d] has no neurons!", layer->index);
        return 0;
    }
    if (layer->index == 0) {
        PSErr(NULL, "Cannot feedforward on layer 0!");
        return 0;
    }
    PSLayer * previous = net->layers[layer->index - 1];
    if (previous == NULL) {
        PSErr(NULL, "Layer[%d]: previous layer is NULL!", layer->index);
        return 0;
    }
    int i, j, x, y, row, col;
    PSLayerParameters * parameters = layer->parameters;
    if (parameters == NULL) {
        PSErr(NULL, "Layer[%d]: parameters are NULL!", layer->index);
        return 0;
    }
    PSLayerParameters * previous_parameters = previous->parameters;
    if (previous_parameters == NULL) {
        PSErr(NULL, "Layer[%d]: parameters are invalid!", layer->index);
        return 0;
    }
    int is_recurrent = (net->flags & FLAG_RECURRENT), times, t;
    if (is_recurrent) {
        va_list args;
        va_start(args, _layer);
        times = va_arg(args, int);
        t = va_arg(args, int);
        va_end(args);
    }
    double * params = parameters->parameters;
    double * previous_params = previous_parameters->parameters;
    int feature_count = (int) (params[PARAM_FEATURE_COUNT]);
    double region_size = params[PARAM_REGION_SIZE];
    double input_w = previous_params[PARAM_OUTPUT_WIDTH];
    double output_w = params[PARAM_OUTPUT_WIDTH];
    int feature_size = size / feature_count;
    int prev_size = previous->size / feature_count;
    for (i = 0; i < feature_count; i++) {
        row = 0;
        col = 0;
        for (j = 0; j < feature_size; j++) {
            int idx = (i * feature_size) + j;
            PSNeuron * neuron = layer->neurons[idx];
            col = idx % (int) output_w;
            if (col == 0 && j > 0) row++;
            int r_row = row * region_size;
            int r_col = col * region_size;
            int max_x = region_size + r_col;
            int max_y = region_size + r_row;
            double max = 0.0, max_z = 0.0;
            for (y = r_row; y < max_y; y++) {
                for (x = r_col; x < max_x; x++) {
                    int nidx = ((y * input_w) + x) + (prev_size * i);
                    PSNeuron * prev_neuron = previous->neurons[nidx];
                    double a = prev_neuron->activation;
                    double z = prev_neuron->z_value;
                    if (a > max) {
                        max = a;
                        max_z = z;
                    }
                }
            }
            neuron->z_value = max_z;
            neuron->activation = max;
#ifdef USE_AVX
            if (!is_recurrent)
                layer->avx_activation_cache[idx] = neuron->activation;
#endif
            if (is_recurrent) {
                PSAddRecurrentState(neuron, neuron->activation, times, t);
                if (neuron->extra == NULL) {
                    PSErr("pool", "Failed to allocate Recurrent Cell!");
                    return 0;
                }
            }
        }
    }
    return 1;
}

/* Backpropagation Functions */

int PSPoolingBackprop(PSLayer * pooling_layer, PSLayer * convolutional_layer,
                      double * delta)
{
    double * new_delta = convolutional_layer->delta;
    PSLayerParameters * pool_params = pooling_layer->parameters;
    PSLayerParameters * conv_params = convolutional_layer->parameters;
    int feature_count = (int) (conv_params->parameters[PARAM_FEATURE_COUNT]);
    int pool_size = (int) (pool_params->parameters[PARAM_REGION_SIZE]);
    int feature_size = pooling_layer->size / feature_count;
    double input_w = pool_params->parameters[PARAM_INPUT_WIDTH];
    double output_w = pool_params->parameters[PARAM_OUTPUT_WIDTH];
    int prev_size = convolutional_layer->size / feature_count;
    int i, j, row, col, x, y;
    for (i = 0; i < feature_count; i++) {
        row = 0;
        col = 0;
        for (j = 0; j < feature_size; j++) {
            int idx = j + (i * feature_size);
            double d = delta[idx];
            PSNeuron * neuron = pooling_layer->neurons[idx];
            col = idx % (int) output_w;
            if (col == 0 && j > 0) row++;
            int r_row = row * pool_size;
            int r_col = col * pool_size;
            int max_x = pool_size + r_col;
            int max_y = pool_size + r_row;
            //double max = 0;
            for (y = r_row; y < max_y; y++) {
                for (x = r_col; x < max_x; x++) {
                    int nidx = ((y * input_w) + x) + (prev_size * i);
                    PSNeuron * prev_neuron = convolutional_layer->neurons[nidx];
                    double a = prev_neuron->activation;
                    new_delta[nidx] = (a < neuron->activation ? 0 : d);
                }
            }
            
        }
    }
    return 1;
}

int PSConvolutionalBackprop(PSLayer* convolutional_layer, PSLayer * prev_layer,
                            PSGradient * lgradients)
{
    double * delta = convolutional_layer->delta;
    int size = convolutional_layer->size;
    PSLayerParameters * params = convolutional_layer->parameters;
    int feature_count = (int) (params->parameters[PARAM_FEATURE_COUNT]);
    int region_size = (int) (params->parameters[PARAM_REGION_SIZE]);
    int stride = (int) (params->parameters[PARAM_STRIDE]);
    double input_w = params->parameters[PARAM_INPUT_WIDTH];
    double output_w = params->parameters[PARAM_OUTPUT_WIDTH];
    int feature_size = size / feature_count;
    int previous_feature_size = 0, prev_features = 1, prev_features_step = 1;
    if (prev_layer->type == Pooling) {
        PSLayerParameters * prev_params = prev_layer->parameters;
        //TODO: check if prev_params == NULL
        prev_features = (int) (prev_params->parameters[PARAM_FEATURE_COUNT]);
        previous_feature_size = prev_layer->size / prev_features;
        prev_features_step = feature_count / prev_features;
    }
    int i, j, row, col, x, y;
    for (i = 0; i < feature_count; i++) {
        PSGradient * feature_gradient = &(lgradients[i]);
        int previous_feature = 0, feature_offset = 0;
        if (prev_features > 1) {
            previous_feature = i / prev_features_step;
            feature_offset = previous_feature * previous_feature_size;
        }
        row = 0;
        col = 0;
        for (j = 0; j < feature_size; j++) {
            int idx = j + (i * feature_size);
            double d = delta[idx];
            feature_gradient->bias += d;
            
            col = idx % (int) output_w;
            if (col == 0 && j > 0) row++;
            int r_row = row * stride;
            int r_col = col * stride;
            int max_x = region_size + r_col;
            int max_y = region_size + r_row;
            int widx = 0;
            for (y = r_row; y < max_y; y++) {
                for (x = r_col; x < max_x; x++) {
                    int nidx = feature_offset + (y * input_w) + x;
                    //printf("  -> %d,%d [%d]\n", x, y, nidx);
                    PSNeuron * prev_neuron = prev_layer->neurons[nidx];
                    double a = prev_neuron->activation;
                    feature_gradient->weights[widx++] += (a * d);
                }
            }
        }
    }
    return 1;
}
