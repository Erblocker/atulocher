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
#include <time.h>

#ifdef USE_AVX
#include "avx.h"
#endif

#include "psyc.h"
#include "utils.h"
#include "convolutional.h"
#include "recurrent.h"
#include "lstm.h"

int PSGlobalFlags = 0;

typedef double (*PSGetDeltaFunction)(PSNeuron* n, PSLayer* l, PSLayer* next,
                                     double * last_d);

static PSLossFunction loss_functions[] = {
    NULL,
    PSQuadraticLoss,
    PSCrossEntropyLoss
};

static size_t loss_functions_count = sizeof(loss_functions) /
                                     sizeof(PSLossFunction);

/* Function Prototypes */

void PSDeleteLayerGradients(PSGradient * lgradients, int size);
void PSDeleteGradients(PSGradient ** gradients, PSNeuralNetwork * network);

/* Feedforward Functions */

static int fullFeedforward(void * _net, void * _layer, ...) {
    PSNeuralNetwork * network = (PSNeuralNetwork*) _net;
    PSLayer * layer = (PSLayer*) _layer;
    int size = layer->size;
    char * func = "fullFeedforward";
    if (layer->neurons == NULL) {
        PSErr(NULL, "Layer[%d] has no neurons!", layer->index);
        return 0;
    }
    if (layer->index == 0) {
        PSErr(NULL, "Cannot feedforward on layer 0!");
        return 0;
    }
    PSLayer * previous = network->layers[layer->index - 1];
    if (previous == NULL) {
        PSErr(NULL, "Layer[%d]: previous layer is NULL!", layer->index);
        return 0;
    }
    int i, j, previous_size = previous->size;
    int is_recurrent = (network->flags & FLAG_RECURRENT), times, t;
    if (is_recurrent) {
        va_list args;
        va_start(args, _layer);
        times = va_arg(args, int);
        t = va_arg(args, int);
        va_end(args);
    }
    for (i = 0; i < size; i++) {
        PSNeuron * neuron = layer->neurons[i];
        double sum = 0.0;
        j = 0;
#ifdef USE_AVX
        AVXDotProduct(previous_size, previous->avx_activation_cache,
                      neuron->weights, sum, j, is_recurrent, t);
#endif
        for (; j < previous_size; j++) {
            PSNeuron * prev_neuron = previous->neurons[j];
            if (prev_neuron == NULL) {
                PSErr(NULL, "Layer[%d]: previous layer's neuron[%d] is NULL!",
                      layer->index, j);
                return 0;
            }
            double a = prev_neuron->activation;
            sum += (a * neuron->weights[j]);
        }
        neuron->z_value = sum + neuron->bias;
        neuron->activation = layer->activate(neuron->z_value);
#ifdef USE_AVX
        if (!is_recurrent)
            layer->avx_activation_cache[i] = neuron->activation;
#endif
        if (is_recurrent) {
            PSAddRecurrentState(neuron, neuron->activation, times, t);
            if (neuron->extra == NULL) {
                PSErr(func, "Failed to allocate Recurrent Cell!");
                return 0;
            }
        }
    }
    return 1;
}

static int softmaxFeedforward(void * _net, void * _layer, ...) {
    PSNeuralNetwork * net = (PSNeuralNetwork*) _net;
    PSLayer * layer = (PSLayer*) _layer;
    int size = layer->size;
    char * func = "softmaxFeedforward";
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
    int i, j, previous_size = previous->size;
    int is_recurrent = (net->flags & FLAG_RECURRENT), times, t;
    if (is_recurrent) {
        va_list args;
        va_start(args, _layer);
        times = va_arg(args, int);
        t = va_arg(args, int);
        va_end(args);
    }
    double max = 0.0, esum = 0.0;
    for (i = 0; i < size; i++) {
        PSNeuron * neuron = layer->neurons[i];
        double sum = 0;
        j = 0;
#ifdef USE_AVX
        AVXDotProduct(previous_size, previous->avx_activation_cache,
                      neuron->weights, sum, j, is_recurrent, t);
#endif
        for (; j < previous_size; j++) {
            PSNeuron * prev_neuron = previous->neurons[j];
            if (prev_neuron == NULL) {
                PSErr(NULL, "Layer[%d]: previous layer's neuron[%d] is NULL!",
                      layer->index, j);
                return 0;
            }
            double a = prev_neuron->activation;
            sum += (a * neuron->weights[j]);
        }
        neuron->z_value = sum + neuron->bias;
        if (i == 0)
            max = neuron->z_value;
        else if (neuron->z_value > max)
            max = neuron->z_value;
    }
    for (i = 0; i < size; i++) {
        PSNeuron * neuron = layer->neurons[i];
        double z = neuron->z_value;
        double e = exp(z - max);
        esum += e;
        neuron->activation = e;
    }
    for (i = 0; i < size; i++) {
        PSNeuron * neuron = layer->neurons[i];
        neuron->activation /= esum;
#ifdef USE_AVX
        if (!is_recurrent)
            layer->avx_activation_cache[i] = neuron->activation;
#endif
        if (is_recurrent) {
            PSAddRecurrentState(neuron, neuron->activation, times, t);
            if (neuron->extra == NULL) {
                PSErr(func, "Failed to allocate Recurrent Cell!");
                return 0;
            }
        }
    }
    return 1;
}

/* Utils */

static double norm(double* matrix, int size) {
    double r = 0.0;
    int i;
    for (i = 0; i < size; i++) {
        double v = matrix[i];
        r += (v * v);
    }
    return sqrt(r);
}

static void shuffle ( double * array, int size, int element_size )
{
    srand ( time(NULL) );
    int byte_size = element_size * sizeof(double);
    for (int i = size - 1; i > 0; i--) {
        int j = rand() % (i+1);
        //printf("Shuffle cycle %d: random is %d\n", i, j);
        double tmp_a[element_size];
        double tmp_b[element_size];
        int idx_a = i * element_size;
        int idx_b = j * element_size;
        //printf("-> idx_a: %d\n", idx_a);
        //printf("-> idx_b: %d\n", idx_b);
        memcpy(tmp_a, array + idx_a, byte_size);
        memcpy(tmp_b, array + idx_b, byte_size);
        memcpy(array + idx_a, tmp_b, byte_size);
        memcpy(array + idx_b, tmp_a, byte_size);
    }
}

static void shuffleSeries ( double ** series, int size)
{
    srand ( time(NULL) );
    for (int i = size - 1; i > 0; i--) {
        int j = rand() % (i+1);
        //printf("Shuffle cycle %d: random is %d\n", i, j);
        double * tmp_a = series[i];
        double * tmp_b = series[j];
        series[i] = tmp_b;
        series[j] = tmp_a;
    }
}

static double ** getRecurrentSeries(double * array, int series_count,
                                    int x_size, int y_size)
{
    double ** series = malloc(series_count * sizeof(double**));
    if (series == NULL) {
        PSErr(NULL, "Could not allocate memory for recurrent series!");
        return NULL;
    }
    int i;
    double * p = array;
    for (i = 0; i < series_count; i++) {
        int series_size = (int) *p;
        if (!series_size) {
            PSErr(NULL, "Invalid series size 0 at %d", (int) (p - array));
            free(series);
            return NULL;
        }
        series[i] = p++;
        p += ((series_size * x_size) + (series_size * y_size));
    }
    return series;
}

static int arrayMaxIndex(double * array, int len) {
    int i;
    double max = 0;
    int max_idx = 0;
    for (i = 0; i < len; i++) {
        double v = array[i];
        if (v > max) {
            max = v;
            max_idx = i;
        }
    }
    return max_idx;
}

/*static double arrayMax(double * array, int len) {
    int i;
    double max = 0;
    for (i = 0; i < len; i++) {
        double v = array[i];
        if (v > max) {
            max = v;
        }
    }
    return max;
}*/

static void fetchRecurrentOutputState(PSLayer * out, double * outputs,
                                      int i, int onehot)
{
    int t = (onehot ? i : i % out->size), j;
    int max_idx = 0;
    double max = 0.0;
    for (j = 0; j < out->size; j++) {
        PSNeuron * neuron = out->neurons[j];
        PSRecurrentCell * cell = GetRecurrentCell(neuron);
        double s = cell->states[t];
        if (onehot) {
            if (s > max) {
                max = s;
                max_idx = j;
            }
        } else {
            outputs[i] = s;
        }
    }
    if (onehot) outputs[i] = max_idx;
}

static double getDeltaForNeuron(PSNeuron * neuron,
                                PSLayer * layer,
                                PSLayer * nextLayer,
                                double * last_delta)
{
    int index = neuron->index, i;
    double dv = 0;
    for (i = 0; i < nextLayer->size; i++) {
        PSNeuron * nextNeuron = nextLayer->neurons[i];
        double weight = nextNeuron->weights[index];
        double d = last_delta[i];
        dv += (d * weight);
    }
    if (layer->derivative != NULL)
        dv *= layer->derivative(neuron->activation);
    return dv;
}

static int compareVersion(const char* vers1, const char* vers2) {
    int major1 = 0, minor1 = 0, patch1 = 0;
    int major2 = 0, minor2 = 0, patch2 = 0;
    sscanf(vers1, "%d.%d.%d", &major1, &minor1, &patch1);
    sscanf(vers2, "%d.%d.%d", &major2, &minor2, &patch2);
    if (major1 < major2) return -1;
    if (major1 > major2) return 1;
    if (minor1 < minor2) return -1;
    if (minor1 > minor2) return 1;
    if (patch1 < patch2) return -1;
    if (patch1 > patch2) return 1;
    return 0;
}

char * PSGetLabelForType(PSLayerType type) {
    switch (type) {
        case FullyConnected:
            return "Fully Connected";
        case Convolutional:
            return "Convolutional";
        case Pooling:
            return "Pooling";
        case Recurrent:
            return "Recurrent";
        case LSTM:
            return "LSTM";
        case SoftMax:
            return "Softmax";
    }
    return "UNKOWN";
}

char * PSGetLayerTypeLabel(PSLayer * layer) {
    return PSGetLabelForType(layer->type);
}

static char * getLossFunctionName(PSLossFunction function) {
    if (function == NULL) return "null";
    if (function == PSQuadraticLoss) return "quadratic";
    else if (function == PSCrossEntropyLoss) return "cross_entropy";
    return "UNKOWN";
}

static char * getNetworkStatusLabel(PSNeuralNetwork * network) {
    if (network == NULL) return "";
    int status = network->status;
    switch (status) {
        case STATUS_UNTRAINED:
            return "untrained";
        case STATUS_TRAINING:
            return "training";
        case STATUS_TRAINED:
            return "trained";
        case STATUS_ERROR:
            return "error";
    }
    return "UNKOWN";
}

static void printLayerInfo(PSLayer * layer) {
    if (layer == NULL) return;
    PSLayerType ltype = layer->type;
    char * type_name = PSGetLayerTypeLabel(layer);
    PSLayerParameters * lparams = layer->parameters;
    char onehot_info[50];
    onehot_info[0] = 0;
    if (layer->index == 0 && layer->flags & FLAG_ONEHOT) {
        PSLayerParameters * params = layer->parameters;
        int onehot_sz = (int) (params->parameters[0]);
        sprintf(onehot_info, " (vector size: %d)", onehot_sz);
    }
    printf("Layer[%d]: %s, size = %d", layer->index, type_name, layer->size);
    if (onehot_info[0]) printf(" %s", onehot_info);
    if ((ltype == Convolutional || ltype == Pooling) && lparams != NULL) {
        double * params = lparams->parameters;
        int fcount = (int) (params[PARAM_FEATURE_COUNT]);
        int rsize = (int) (params[PARAM_REGION_SIZE]);
        int input_w = (int) (params[PARAM_INPUT_WIDTH]);
        int input_h = (int) (params[PARAM_INPUT_HEIGHT]);
        int stride = (int) (params[PARAM_STRIDE]);
        int use_relu = (int) (params[PARAM_USE_RELU]);
        char * actv = (use_relu ? "relu" : "sigmoid");
        printf(", input size = %dx%d, features = %d", input_w, input_h, fcount);
        printf(", region = %dx%d, stride = %d, activation = %s\n",
               rsize, rsize, stride, actv);
    } else printf("\n");
}

void PSPrintNetworkInfo(PSNeuralNetwork * network) {
    if (network == NULL) return;
    const char * name = network->name;
    if (name == NULL || !strlen(name)) name = "UNNAMED NETWORK";
    printf("Network: %s\n", name);
    printf("Size: %d\nLayers:\n", network->size);
    int i;
    for (i = 0; i < network->size; i++) {
        PSLayer * layer = network->layers[i];
        printLayerInfo(layer);
    }
    char * loss_name = getLossFunctionName(network->loss);
    printf("Loss Function: %s\n", loss_name);
    printf("Status: %s\n", getNetworkStatusLabel(network));
}

/* Loss Functions */

double PSQuadraticLoss(double * outputs, double * desired, int size,
                     int onehot_size)
{
    double * _diffs;
    double diffs[size];
    if (!onehot_size) {
        int i;
        for (i = 0; i < size; i++) {
            double d = outputs[i] - desired[i];
            diffs[i] = d;
        }
        _diffs = diffs;
    } else _diffs = outputs;
    double n = norm(_diffs, size);
    double loss = 0.5 * (n * n);
    if (onehot_size) loss /= (double) onehot_size;
    return loss;
}

double PSCrossEntropyLoss(double * outputs, double * desired, int size,
                        int onehot_size)
{
    double loss = 0.0;
    int i;
    for (i = 0; i < size; i++) {
        double o = outputs[i];
        if (o == 0.0) continue;
        if (onehot_size) loss += (log(o));
        else {
            if (o == 1) continue; // log(1 - 1) would be NaN
            double y = desired[i];
            loss += (y * log(o) + (1 - y) * log(1 - o));
        }
    }
    loss *= -1;
    if (onehot_size)
        loss = (loss / (double) size);// / log((double) onehot_size);
    return loss;
}

/* Neural Network Functions */

PSNeuralNetwork * PSCreateNetwork(const char* name) {
    PSNeuralNetwork *network = (malloc(sizeof(PSNeuralNetwork)));
    if (network == NULL) {
        PSErr("PSCreateNetwork", "Could not allocate memory for Network!");
        return NULL;
    }
    network->name = name;
    network->size = 0;
    network->layers = NULL;
    network->input_size = 0;
    network->output_size = 0;
    network->status = STATUS_UNTRAINED;
    network->current_epoch = 0;
    network->current_batch = 0;
    network->flags = FLAG_NONE;
    network->loss = PSQuadraticLoss;
    network->onEpochTrained = NULL;
    return network;
}

PSNeuralNetwork * PSCloneNetwork(PSNeuralNetwork * network, int layout_only) {
    if (network == NULL) return NULL;
    PSNeuralNetwork * clone = PSCreateNetwork(NULL);
    if (clone == NULL) return NULL;
    char * func = "PSCloneNetwork";
    if (!layout_only) {
        clone->status = network->status;
        clone->current_epoch = network->current_epoch;
        clone->current_batch = network->current_batch;
    }
    clone->flags = network->flags;
    clone->loss = network->loss;
    
    int i, j, k, w;
    for (i = 0; i < network->size; i++) {
        PSLayer * layer = network->layers[i];
        PSLayerType type = layer->type;
        PSLayerParameters * oparams = layer->parameters;
        PSLayerParameters * cparams = NULL;
        if (oparams) {
            cparams = malloc(sizeof(PSLayerParameters));
            if (cparams == NULL) {
                PSErr(func, "Layer[%d]: Could not allocate layer params!", i);
                PSDeleteNetwork(clone);
                return NULL;
            }
            cparams->count = oparams->count;
            cparams->parameters = malloc(cparams->count * sizeof(double));
            if (cparams->parameters == NULL) {
                PSErr(func, "Layer[%d]: Could not allocate layer params!", i);
                PSDeleteNetwork(clone);
                return NULL;
            }
            for (j = 0; j < cparams->count; j++)
                cparams->parameters[j] = oparams->parameters[j];
        }
        PSLayer * cloned_layer = PSAddLayer(clone, type, layer->size, cparams);
        if (cloned_layer == NULL) {
            PSDeleteNetwork(clone);
            return NULL;
        }
        cloned_layer->flags = layer->flags;
        if (!layout_only) {
            void * extra = layer->extra;
            if (Convolutional == type && extra) {
                PSSharedParams * oshared = getConvSharedParams(layer);
                PSSharedParams * cshared = getConvSharedParams(cloned_layer);
                cshared->feature_count = oshared->feature_count;
                cshared->weights_size = oshared->weights_size;
                for (k = 0; k < cshared->feature_count; k++) {
                    cshared->biases[k] = oshared->biases[k];
                    for (w = 0; w < cshared->weights_size; w++)
                        cshared->weights[k][w] = oshared->weights[k][w];
                }
            }
            for (j = 0; j < layer->size; j++) {
                PSNeuron * orig_n = layer->neurons[j];
                PSNeuron * clone_n = cloned_layer->neurons[j];
                clone_n->activation = orig_n->activation;
                clone_n->z_value = orig_n->z_value;
                //if (Pooling == type) continue;
                clone_n->bias = orig_n->bias;

                if (Convolutional != type && Pooling != type) {
                    double * oweights = orig_n->weights;
                    double * cweights = clone_n->weights;
                    for (w = 0; w < orig_n->weights_size; w++)
                        cweights[w] = oweights[w];
                }
                if (layer->flags & FLAG_RECURRENT) {
                    PSRecurrentCell * ocell = GetRecurrentCell(orig_n);
                    PSRecurrentCell * ccell = GetRecurrentCell(clone_n);
                    int sc = ocell->states_count;
                    ccell->states_count = sc;
                    if (sc > 0) {
                        ccell->states = malloc(sc * sizeof(double));
                        if (ccell->states == NULL) {
                            printMemoryErrorMsg();
                            PSDeleteNetwork(clone);
                            return NULL;
                        }
                        for (k = 0; k < sc; k++)
                            ccell->states[k] = ocell->states[k];
                    }
                }
                if (layer->type == LSTM) {
                    PSLSTMCell * ocell = GetLSTMCell(orig_n);
                    PSLSTMCell * ccell = GetLSTMCell(clone_n);
                    ccell->candidate_bias = ocell->candidate_bias;
                    ccell->input_bias = ocell->input_bias;
                    ccell->output_bias = ocell->output_bias;
                    ccell->forget_bias = ocell->forget_bias;
                }
            }
        }
    }
    return clone;
}

int PSLoadNetwork(PSNeuralNetwork * network, const char* filename) {
    if (network == NULL) return 0;
    FILE * f = fopen(filename, "r");
    printf("Loading network from %s\n", filename);
    if (f == NULL) {
        fprintf(stderr, "Cannot open %s!\n", filename);
        return 0;
    }
    char * func = "PSLoadNetwork";
    int netsize, i, j, k;
    int empty = (network->size == 0);
    char vers[20] = "0.0.0";
    int v0 = 0, v1 = 0, v2 = 0;
    int epochs = 0, batch_count = 0;
    int matched = fscanf(f, "--v%d.%d.%d", &v0, &v1, &v2);
    if (matched) {
        sprintf(vers, "%d.%d.%d", v0, v1, v2);
        printf("File version is %s (current: %s).\n", vers, PSYC_VERSION);
        int idx = 0, val = 0;
        PSLossFunction loss = NULL;
        while ((matched = fscanf(f, ",%d", &val))) {
            switch (idx++) {
                case 0:
                    network->flags |= val; break;
                case 1:
                    if ((size_t) val < loss_functions_count) {
                        loss = loss_functions[val];
                        network->loss = loss;
                        printf("Loss Function: %s\n",getLossFunctionName(loss));
                    }
                    break;
                case 2: epochs = val; break;
                case 3: batch_count = val; break;
                default:
                    break;
            }
        }
        fscanf(f, "\n");
    }
    matched = fscanf(f, "%d:", &netsize);
    if (!matched) {
        PSErr(func, "Invalid file %s!", filename);
        fclose(f);
        return 0;
    }
    if (!empty && network->size != netsize) {
        PSErr(func, "Network size differs!");
        fclose(f);
        return 0;
    }
    char sep[] = ",";
    char eol[] = "\n";
    int min_argc = (compareVersion(vers, "0.0.0") == 1 ? 2 : 1);
    PSLayer * layer = NULL;
    for (i = 0; i < netsize; i++) {
        int lsize = 0;
        int lflags = 0;
        PSLayerType ltype = FullyConnected;
        int args[20];
        int argc = 0, aidx = 0;
        char * last = (i == (netsize - 1) ? eol : sep);
        char fmt[50];
        char buff[255];
        sprintf(fmt, "%%d%s", last);
        //fputs(fmt, stderr);
        matched = fscanf(f, fmt, &lsize);
        if (!matched) {
            int type = 0, arg = 0;
            argc = 0;
            matched = fscanf(f, "[%d,%d", &type, &argc);
            if (!matched) {
                PSErr(func, "Invalid header: layer[%d], col. %ld!",
                      i, ftell(f));
                fclose(f);
                return 0;
            }
            if (argc == 0) {
                PSErr(func, "Layer must have at least 1 argument (size)");
                fclose(f);
                return 0;
            }
            ltype = (PSLayerType) type;
            for (aidx = 0; aidx < argc; aidx++) {
                matched = fscanf(f, ",%d", &arg);
                if (!matched) {
                    PSErr(func, "Invalid header: l%d, arg. %d, col. %ld!",
                          i, aidx, ftell(f));
                    fclose(f);
                    return 0;
                }
                if (aidx == 0) lsize = arg;
                else if (min_argc > 1 && aidx == 1) lflags = arg;
                else args[aidx - min_argc] = arg;
            }
            argc -= min_argc;
            sprintf(fmt, "]%s", last);
            fscanf(f, fmt, buff);
        }
        if (!empty) {
            layer = network->layers[i];
            if (layer->size != lsize) {
                PSErr(func, "Layer %d size %d differs from %d!", i,
                      layer->size, lsize);
                fclose(f);
                return 0;
            }
            if (ltype != layer->type) {
                PSErr(func, "Layer %d type %d differs from %d!", i,
                      (int) (layer->type), (int) ltype);
                fclose(f);
                return 0;
            }
            if (ltype == Convolutional || ltype == Pooling) {
                PSLayerParameters * params = layer->parameters;
                if (params == NULL) {
                    PSErr(func, "Layer %d params are NULL!", i);
                    fclose(f);
                    return 0;
                }
                for (aidx = 0; aidx < argc; aidx++) {
                    if (aidx >= params->count) break;
                    int arg = args[aidx];
                    double val = params->parameters[aidx];
                    if (arg != (int) val) {
                        PSErr(func, "Layer %d arg[%d] %d diff. from %d!",
                              i, aidx,(int) val, arg);
                        fclose(f);
                        return 0;
                    }
                }
            }
        } else {
            layer = NULL;
            PSLayerParameters * params = NULL;
            if (ltype == Convolutional || ltype == Pooling) {
                int param_c = CONV_PARAMETER_COUNT;
                params = PSCreateLayerParamenters(param_c);
                for (aidx = 0; aidx < argc; aidx++) {
                    if (aidx >= param_c) break;
                    int arg = args[aidx];
                    params->parameters[aidx] = (double) arg;
                }
                layer = PSAddLayer(network, ltype, lsize, params);
            } else {
                if (network->size == 0 && (lflags & FLAG_ONEHOT) && argc > 0) {
                    lsize = args[0];
                    network->flags |= FLAG_ONEHOT;
                } else if (argc > 0) {
                    params = PSCreateLayerParamenters(argc);
                    for (aidx = 0; aidx < argc; aidx++) {
                        int arg = args[aidx];
                        params->parameters[aidx] = (double) arg;
                    }
                }
                layer = PSAddLayer(network, ltype, lsize, params);
            }
            if (layer == NULL) {
                PSErr(func, "Could not create layer %d", i);
                fclose(f);
                return 0;
            }
            layer->flags |= lflags;
        }
    }
    for (i = 1; i < network->size; i++) {
        layer = network->layers[i];
        int lsize = 0;
        PSSharedParams * shared = NULL;
        if (layer->type == Convolutional) {
            shared = getConvSharedParams(layer);
            if (shared == NULL) {
                PSErr(func, "Layer %d, missing shared params!", i);
                fclose(f);
                return 0;
            }
            lsize = shared->feature_count;
        } else if (layer->type == Pooling) {
            continue;
        } else lsize = layer->size;
        int is_lstm = (LSTM == layer->type);
        for (j = 0; j < lsize; j++) {
            double bias = 0;
            int wsize = 0;
            double * weights = NULL;
            //LSTM biases
            double cb = 0.0, ib = 0.0, ob = 0.0, fb = 0.0;
            if (!is_lstm)
                matched = fscanf(f, "%lf|", &bias);
            else
                matched = fscanf(f, "%lf,%lf,%lf,%lf|", &cb, &ib, &ob, &fb);
            if (!matched) {
                PSErr(func, "Layer %d, neuron %d: invalid bias!", i, j);
                fclose(f);
                return 0;
            }
            if (shared == NULL) {
                PSNeuron * neuron = layer->neurons[j];
                wsize = neuron->weights_size;
                neuron->bias = bias;
                weights = neuron->weights;
                if (is_lstm) {
                    PSLSTMCell * cell = GetLSTMCell(neuron);
                    assert(cell != NULL);
                    cell->candidate_bias = cb;
                    cell->input_bias = ib;
                    cell->output_bias = ob;
                    cell->forget_bias = fb;
                }
            } else {
                shared->biases[j] = bias;
                wsize = shared->weights_size;
                weights = shared->weights[j];
            }
            for (k = 0; k < wsize; k++) {
                double w = 0;
                char * last = (k == (wsize - 1) ? eol : sep);
                char fmt[5];
                sprintf(fmt, "%%lf%s", last);
                matched = fscanf(f, fmt, &w);
                if (!matched) {
                    PSErr(func,"Layer %d neuron %d: invalid weight[%d]",
                          i, j, k);
                    fclose(f);
                    return 0;
                }
                weights[k] = w;
                printf("\rLoading layer %d, neuron %d   ", i, j);
                fflush(stdout);
            }
        }
    }
    printf("\n");
    fclose(f);
    return 1;
}

int PSSaveNetwork(PSNeuralNetwork * network, const char* filename) {
    char * func = "saveNetwork";
    if (network->size == 0) {
        PSErr(func, "Empty network!");
        return 0;
    }
    FILE * f = fopen(filename, "w");
    printf("Saving network to %s\n", filename);
    if (f == NULL) {
        fprintf(stderr, "Cannot open %s for writing!\n", filename);
        return 0;
    }
    int i, j, k, loss_function = 0;
    // Header
    fprintf(f, "--v%s", PSYC_VERSION);
    for (i = 0; i < (int) loss_functions_count; i++) {
        if (network->loss == loss_functions[i]) {
            loss_function = i;
            break;
        }
    }
    fprintf(f, ",%d,%d,%d,%d\n", network->flags, loss_function,
            network->current_epoch, network->current_batch);
    
    fprintf(f, "%d:", network->size);
    for (i = 0; i < network->size; i++) {
        PSLayer * layer = network->layers[i];
        PSLayerType ltype = layer->type;
        if (i > 0) fprintf(f, ",");
        int flags = layer->flags;
        PSLayerParameters * params = layer->parameters;
        if (FullyConnected == ltype && !flags && !params)
            fprintf(f, "%d", layer->size);
        else if (params) {
            int argc = params->count;
            fprintf(f, "[%d,%d,%d,%d", (int) ltype, 2 + argc, layer->size,
                    layer->flags);
            for (j = 0; j < argc; j++) {
                fprintf(f, ",%d", (int) (params->parameters[j]));
            }
            fprintf(f, "]");
        } else {
            fprintf(f, "[%d,2,%d,%d]", (int) ltype, layer->size, flags);
        }
    }
    fprintf(f, "\n");
    for (i = 1; i < network->size; i++) {
        PSLayer * layer = network->layers[i];
        PSLayerType ltype = layer->type;
        int lsize = layer->size;
        if (Convolutional == ltype) {
            PSSharedParams * shared = getConvSharedParams(layer);
            if (shared == NULL) {
                PSErr(func, "Layer[%d]: shared params are NULL!", i);
                fclose(f);
                return 0;
            }
            int feature_count = shared->feature_count;
            if (feature_count < 1) {
                PSErr(func, "Layer[%d]: feature count must be >= 1!", i);
                fclose(f);
                return 0;
            }
            for (j = 0; j < feature_count; j++) {
                double bias = shared->biases[j];
                double * weights = shared->weights[j];
                fprintf(f, "%.15e|", bias);
                for (k = 0; k < shared->weights_size; k++) {
                    if (k > 0) fprintf(f, ",");
                    double w = weights[k];
                    fprintf(f, "%.15e", w);
                }
                fprintf(f, "\n");
            }
        }
        else if (Pooling == ltype) continue;
        else {
            int is_lstm = (LSTM == ltype);
            for (j = 0; j < lsize; j++) {
                PSNeuron * neuron = layer->neurons[j];
                if (!is_lstm)
                    fprintf(f, "%.15e|", neuron->bias);
                else {
                    PSLSTMCell * cell = GetLSTMCell(neuron);
                    assert(cell != NULL);
                    fprintf(f, "%.15e,%.15e,%.15e,%.15e|",
                            cell->candidate_bias,
                            cell->input_bias,
                            cell->output_bias,
                            cell->forget_bias);
                }
                for (k = 0; k < neuron->weights_size; k++) {
                    if (k > 0) fprintf(f, ",");
                    double w = neuron->weights[k];
                    fprintf(f, "%.15e", w);
                }
                fprintf(f, "\n");
            }
        }
    }
    fclose(f);
    return 1;
}

void PSDeleteNetwork(PSNeuralNetwork * network) {
    int size = network->size;
    int i, is_recurrent = (network->flags & FLAG_RECURRENT);
    for (i = 0; i < size; i++) {
        PSLayer * layer = network->layers[i];
        if (is_recurrent) layer->flags |= FLAG_RECURRENT;
        PSDeleteLayer(layer);
    }
    free(network->layers);
    free(network);
}

void PSDeleteNeuron(PSNeuron * neuron, PSLayer * layer) {
    if (neuron->weights != NULL) free(neuron->weights);
    if (neuron->extra != NULL) {
        if (layer->flags & FLAG_RECURRENT) {
            if (layer->type == LSTM)
                PSDeleteLSTMCell(GetLSTMCell(neuron));
            else {
                PSRecurrentCell * cell = GetRecurrentCell(neuron);
                if (cell->states != NULL) free(cell->states);
                free(cell);
            }
        } else free(neuron->extra);
    }
    free(neuron);
}

PSLayer * PSAddLayer(PSNeuralNetwork * network, PSLayerType type, int size,
                     PSLayerParameters* params) {
    if (network == NULL) return NULL;
    char * func = "PSAddLayer";
    if (network->size == 0 && type != FullyConnected) {
        PSErr(func, "First layer type must be FullyConnected");
        return NULL;
    }
    PSLayer * layer = malloc(sizeof(PSLayer));
    if (layer == NULL) {
        PSErr(func, "Could not allocate layer %d!", network->size);
        return NULL;
    }
    layer->network = network;
    layer->index = network->size++;
    layer->type = type;
    layer->size = size;
    layer->parameters = params;
    layer->extra = NULL;
    layer->flags = FLAG_NONE;
    layer->delta = NULL;
#ifdef USE_AVX
    layer->avx_activation_cache = NULL;
#endif
    PSLayer * previous = NULL;
    int previous_size = 0;
    int initialized = 0;
    //printf("Adding layer %d\n", layer->index);
    if (network->layers == NULL) {
        network->layers = malloc(sizeof(PSLayer*));
        if (network->layers == NULL) {
            PSAbortLayer(network, layer);
            PSErr(func, "Could not allocate network layers!");
            return NULL;
        }
        if ((network->flags & FLAG_ONEHOT) && params == NULL) {
            layer->flags |= FLAG_ONEHOT;
            PSLayerParameters * params;
            params = PSCreateLayerParamenters(1, (double) size);
            layer->parameters = params;
            size = 1;
            layer->size = 1;
        }
        network->input_size = size;
    } else {
        network->layers = realloc(network->layers,
                                  sizeof(PSLayer*) * network->size);
        if (network->layers == NULL) {
            PSAbortLayer(network, layer);
            PSErr(func, "Could not reallocate network layers!");
            return NULL;
        }
        previous = network->layers[layer->index - 1];
        if (previous == NULL) {
            PSAbortLayer(network, layer);
            PSErr(func, "Previous layer is NULL!");
            return NULL;
        }
        previous_size = previous->size;
        if (layer->index == 1 && previous->flags & FLAG_ONEHOT) {
            PSLayerParameters * params = previous->parameters;
            if (params == NULL) {
                PSAbortLayer(network, layer);
                PSErr(func, "Missing layer params on onehot layer[0]!");
                return NULL;
            }
            previous_size = (int) (params->parameters[0]);
        }
        network->output_size = size;
    }
    if (previous && previous->type == Convolutional && type != Pooling) {
        PSErr(func, "Layer[%d]: only Pooling type is allowd after a "
              "Convolutional layer (type = %s)", layer->index,
              PSGetLabelForType(type));
        PSAbortLayer(network, layer);
        return NULL;
    }
    if (type == FullyConnected || type == SoftMax) {
        layer->neurons = malloc(sizeof(PSNeuron*) * size);
        if (layer->neurons == NULL) {
            PSErr(func, "Layer[%d]: could not allocate neurons!", layer->index);
            PSAbortLayer(network, layer);
            return NULL;
        }
#ifdef USE_AVX
        layer->avx_activation_cache = calloc(size, sizeof(double));
        if (layer->avx_activation_cache == NULL) {
            printMemoryErrorMsg();
            PSAbortLayer(network, layer);
            return NULL;
        }
#endif
        int i, j;
        for (i = 0; i < size; i++) {
            PSNeuron * neuron = malloc(sizeof(PSNeuron));
            if (neuron == NULL) {
                PSAbortLayer(network, layer);
                PSErr(func, "Could not allocate neuron!");
                return NULL;
            }
            neuron->index = i;
            neuron->extra = NULL;
            if (layer->index > 0) {
                neuron->weights_size = previous_size;
                neuron->bias = gaussian_random(0, 1);
                neuron->weights = malloc(sizeof(double) * previous_size);
                for (j = 0; j < previous_size; j++) {
                    neuron->weights[j] = gaussian_random(0, 1);
                }
            } else {
                neuron->bias = 0;
                neuron->weights_size = 0;
                neuron->weights = NULL;
            }
            neuron->activation = 0;
            neuron->z_value = 0;
            neuron->layer = layer;
            layer->neurons[i] = neuron;
        }
        if (type != SoftMax) {
            layer->activate = sigmoid;
            layer->derivative = sigmoid_derivative;
            layer->feedforward = fullFeedforward;
        } else {
            layer->activate = NULL;
            layer->derivative = NULL;
            layer->feedforward = softmaxFeedforward;
            //network->loss = PSCrossEntropyLoss;
        }
        initialized = 1;
    } else if (type == Convolutional) {
        initialized = PSInitConvolutionalLayer(network, layer, params);
    } else if (type == Pooling) {
        initialized = PSInitPoolingLayer(network, layer, params);
    } else if (type == Recurrent) {
        initialized = PSInitRecurrentLayer(network, layer, size, previous_size);
        if (initialized) network->loss = PSCrossEntropyLoss;
    } else if (type == LSTM) {
        initialized = PSInitLSTMLayer(network, layer, size, previous_size);
        if (initialized) network->loss = PSCrossEntropyLoss;
    }
    if (!initialized) {
        PSAbortLayer(network, layer);
        PSErr(func, "Could not initialize layer %d!", network->size + 1);
        return NULL;
    }
    if (layer->index > 0) {
        int dsize = layer->size;
        if (type == LSTM) dsize *= 2;
        layer->delta = calloc(dsize, sizeof(double));
        if (layer->delta == NULL) {
            PSAbortLayer(network, layer);
            printMemoryErrorMsg();
            PSErr(func, "Could not initialize layer %d!", network->size + 1);
            return NULL;
        }
    }
    network->layers[layer->index] = layer;
    printLayerInfo(layer);
    return layer;
}

PSLayer * PSAddConvolutionalLayer(PSNeuralNetwork * network,
                                  PSLayerParameters* params)
{
    return PSAddLayer(network, Convolutional, 0, params);
}

PSLayer * PSAddPoolingLayer(PSNeuralNetwork * network,
                            PSLayerParameters* params)
{
    return PSAddLayer(network, Pooling, 0, params);
}

void PSDeleteLayer(PSLayer* layer) {
    int size = layer->size;
    int i;
    for (i = 0; i < size; i++) {
        PSNeuron* neuron = layer->neurons[i];
        if (layer->type != Convolutional)
            PSDeleteNeuron(neuron, layer);
        else
            free(neuron);
    }
    free(layer->neurons);
    PSLayerParameters * params = layer->parameters;
    if (params != NULL) PSDeleteLayerParamenters(params);
    void * extra = layer->extra;
    if (extra != NULL) {
        if (layer->type == Convolutional) {
            PSSharedParams * shared = (PSSharedParams*) extra;
            int fc = shared->feature_count;
            //int ws = shared->weights_size;
            if (shared->biases != NULL) free(shared->biases);
            if (shared->weights != NULL) {
                int i;
                for (i = 0; i < fc; i++) free(shared->weights[i]);
                free(shared->weights);
            }
            free(extra);
        } else free(extra);
    }
#ifdef USE_AVX
    if (layer->avx_activation_cache != NULL) free(layer->avx_activation_cache);
#endif
    if (layer->delta != NULL) free(layer->delta);
    free(layer);
}

PSLayerParameters * PSCreateLayerParamenters(int count, ...) {
    PSLayerParameters * params = malloc(sizeof(PSLayerParameters));
    if (params == NULL) {
        PSErr(NULL, "Could not allocate Layer Parameters!");
        return NULL;
    }
    params->count = count;
    if (count == 0) params->parameters = NULL;
    else {
        params->parameters = malloc(sizeof(double) * count);
        if (params->parameters == NULL) {
            PSErr(NULL, "Could not allocate Layer Parameters!");
            free(params);
            return NULL;
        }
        va_list args;
        va_start(args, count);
        int i;
        for (i = 0; i < count; i++)
            params->parameters[i] = va_arg(args, double);
        va_end(args);
    }
    return params;
}

PSLayerParameters * PSCreateConvolutionalParameters(double feature_count,
                                                    double region_size,
                                                    int stride,
                                                    int padding,
                                                    int use_relu)
{
    return PSCreateLayerParamenters(CONV_PARAMETER_COUNT, feature_count,
                                    region_size, (double) stride,
                                    0.0f, 0.0f, 0.0f, 0.0f,
                                    (double) padding, (double) use_relu);
}

int PSSetLayerParameter(PSLayerParameters * params, int param, double value) {
    if (params->parameters == NULL) {
        int len = param + 1;
        params->parameters = malloc(sizeof(double) * len);
        if (params->parameters == NULL) {
            printMemoryErrorMsg();
            return 0;
        }
        memset(params->parameters, 0.0f, sizeof(double) * len);
        params->count = len;
    } else if (param >= params->count) {
        int len = params->count;
        int new_len = param + 1;
        double * old_params = params->parameters;
        size_t size = sizeof(double) * new_len;
        params->parameters = malloc(sizeof(double) * size);
        if (params->parameters == NULL) {
            printMemoryErrorMsg();
            return 0;
        }
        memset(params->parameters, 0.0f, sizeof(double) * size);
        memcpy(params->parameters, old_params, len * sizeof(double));
        free(old_params);
    }
    params->parameters[param] = value;
    return 1;
}

int PSAddLayerParameter(PSLayerParameters * params, double val) {
    return PSSetLayerParameter(params, params->count + 1, val);
}

void PSDeleteLayerParamenters(PSLayerParameters * params) {
    if (params == NULL) return;
    if (params->parameters != NULL) free(params->parameters);
    free(params);
}

int feedforwardThroughTime(PSNeuralNetwork * network, double * values,
                           int times)
{
    if (network == NULL) return 0;
    PSLayer * first = network->layers[0];
    int input_size = first->size;
    char * func = "feedforwardThroughTime";
    int i, t;
    for (t = 0; t < times; t++) {
        for (i = 0; i < input_size; i++) {
            PSNeuron * neuron = first->neurons[i];
            neuron->activation = values[i];
            PSAddRecurrentState(neuron, values[i], times, t);
            if (neuron->extra == NULL) {
                PSErr(func, "Failed to allocate Recurrent Cell!");
                return 0;
            }
        }
        for (i = 1; i < network->size; i++) {
            PSLayer * layer = network->layers[i];
            if (layer == NULL) {
                PSErr(func, "Layer %d is NULL", i);
                return 0;
            }
            if (layer->feedforward == NULL) {
                PSErr(func, "Layer %d feedforward function is NULL", i);
                return 0;
            }
            int ok = layer->feedforward(network, layer, times, t);
            if (!ok) return 0;
        }
        values += input_size;
    }
    return 1;
}

int PSFeedforward(PSNeuralNetwork * network, double * values) {
    if (network == NULL) return 0;
    char * func = "PSFeedforward";
    if (network->size == 0) {
        PSErr(func, "Empty network!");
        return 0;
    }
    if (network->flags & FLAG_RECURRENT) {
        int times = (int) values[0];
        if (times <= 0) {
            PSErr(func, "Recurrent times must be > 0 (found %d)", times);
            return 0;
        }
        return feedforwardThroughTime(network, values + 1, times);
    }
    PSLayer * first = network->layers[0];
    int input_size = first->size;
    int i;
    for (i = 0; i < input_size; i++) {
        first->neurons[i]->activation = values[i];
#ifdef USE_AVX
        first->avx_activation_cache[i] = values[i];
#endif
    }
    for (i = 1; i < network->size; i++) {
        PSLayer * layer = network->layers[i];
        if (layer == NULL) {
            PSErr(func, "Layer %d is NULL!", i);
            return 0;
        }
        if (layer->feedforward == NULL) {
            PSErr(func, "Layer %d feedforward function is NULL", i);
            return 0;
        }
        int success = layer->feedforward(network, layer);
        if (!success) return 0;
    }
    return 1;
}

PSGradient * createLayerGradients(PSLayer * layer) {
    if (layer == NULL) return NULL;
    PSGradient * gradients;
    char * func = "createLayerGradients";
    PSLayerType ltype = layer->type;
    if (ltype == Pooling) return NULL;
    int size = layer->size;
    PSLayerParameters * parameters = NULL;
    if (ltype == Convolutional) {
        parameters = layer->parameters;
        if (parameters == NULL) {
            PSErr(func, "Layer %d parameters are NULL!", layer->index);
            return NULL;
        }
        size = (int) (parameters->parameters[PARAM_FEATURE_COUNT]);
    }
    gradients = malloc(sizeof(PSGradient) * size);
    if (gradients == NULL) {
        PSErr(func, "Could not allocate memory!");
        return NULL;
    }
    int i, ws = 0;
    for (i = 0; i < size; i++) {
        PSNeuron * neuron = layer->neurons[i];
        if (ltype == Convolutional) {
            if (!ws) {
                int region_size =
                    (int) (parameters->parameters[PARAM_REGION_SIZE]);
                ws = region_size * region_size;
            }
        } else {
            ws = neuron->weights_size;
            if (ltype == LSTM) ws += 4; // Make room for LSTM biases
        }
        gradients[i].bias = 0;
        int memsize = sizeof(double) * ws;
        gradients[i].weights = malloc(memsize);
        if (gradients[i].weights == NULL) {
            PSErr(func, "Could not allocate memory!");
            PSDeleteLayerGradients(gradients, size);
            return NULL;
        }
        memset(gradients[i].weights, 0, memsize);
    }
    return gradients;
}

int PSClassify(PSNeuralNetwork * network, double * values) {
    int ok = PSFeedforward(network, values);
    if (!ok) {
        PSErr("PSClassify", "Feedforward failed");
        return -1;
    };
    int netsize = network->size, outsize = network->output_size, i;
    PSLayer * out = network->layers[netsize - 1];
    double max = 0.0;
    int max_idx = 0;
    for (i = 0; i < outsize; i++) {
        double a = out->neurons[i]->activation;
        if (a > max) {
            max = a;
            max_idx = i;
        }
    }
    return max_idx;
}

PSGradient ** createGradients(PSNeuralNetwork * network) {
    if (network == NULL) return NULL;
    PSGradient ** gradients = malloc(sizeof(PSGradient*) * network->size - 1);
    if (gradients == NULL) {
        printMemoryErrorMsg();
        return NULL;
    }
    int i;
    for (i = 1; i < network->size; i++) {
        PSLayer * layer = network->layers[i];
        int idx = i - 1;
        gradients[idx] = createLayerGradients(layer);
        if (gradients[idx] == NULL && layer->type != Pooling) {
            printMemoryErrorMsg();
            PSDeleteGradients(gradients, network);
            return NULL;
        }
    }
    return gradients;
}

void PSDeleteLayerGradients(PSGradient * gradient, int size) {
    int i;
    for (i = 0; i < size; i++) {
        PSGradient g = gradient[i];
        free(g.weights);
    }
    free(gradient);
}

void PSDeleteGradients(PSGradient ** gradients, PSNeuralNetwork * network) {
    int i;
    for (i = 1; i < network->size; i++) {
        PSGradient * lgradients = gradients[i - 1];
        if (lgradients == NULL) continue;
        PSLayer * layer = network->layers[i];
        int lsize;
        if (layer->type == Convolutional) {
            PSLayerParameters * params = layer->parameters;
            lsize = (int) (params->parameters[PARAM_FEATURE_COUNT]);
        } else lsize = layer->size;
        PSDeleteLayerGradients(lgradients, lsize);
    }
    free(gradients);
}

PSGradient ** backprop(PSNeuralNetwork * network, double * x, double * y) {
    if (network == NULL) return NULL;
    PSGradient ** gradients = createGradients(network);
    if (gradients == NULL) return NULL;
    int netsize = network->size;
    PSLayer * outputLayer = network->layers[netsize - 1];
    int osize = outputLayer->size;
    PSGradient * lgradients = gradients[netsize - 2]; //No gradient for inputs
    PSLayer * previousLayer = network->layers[outputLayer->index - 1];
    PSLayer * nextLayer = NULL;
    double * delta = outputLayer->delta;
    double * last_delta = delta;

    int i, o, w, j, ok = 1;
    if (x != NULL) {
        ok = PSFeedforward(network, x);
        if (!ok) {
            PSDeleteGradients(gradients, network);
            return NULL;
        }
    }
    int apply_derivative = shouldApplyDerivative(network);
    double softmax_sum = 0.0;
    for (o = 0; o < osize; o++) {
        PSNeuron * neuron = outputLayer->neurons[o];
        double o_val = neuron->activation;
        double y_val = y[o];
        double d = 0.0;
        if (outputLayer->type != SoftMax) {
            d = o_val - y_val;
            if (apply_derivative)
                d *= outputLayer->derivative(neuron->activation);
        } else {
            y_val = (y_val < 1 ? 0 : 1);
            d = -(y_val - o_val);
            if (apply_derivative) d *= o_val;
            softmax_sum += d;
        }
        delta[o] = d;
        if (outputLayer->type != SoftMax) {
            PSGradient * gradient = &(lgradients[o]);
            gradient->bias = d;
            int wsize = neuron->weights_size;
            w = 0;
#ifdef USE_AVX
            AVXMultiplyValue(wsize, previousLayer->avx_activation_cache, d,
                             gradient->weights, w, 0, 0, 0);
#endif
            for (; w < wsize; w++) {
                double prev_a = previousLayer->neurons[w]->activation;
                gradient->weights[w] = d * prev_a;
            }
        }
    }
    if (outputLayer->type == SoftMax) {
        for (o = 0; o < osize; o++) {
            PSNeuron * neuron = outputLayer->neurons[o];
            double o_val = neuron->activation;
            if (apply_derivative) delta[o] -= (o_val * softmax_sum);
            double d = delta[o];
            PSGradient * gradient = &(lgradients[o]);
            gradient->bias = d;
            int wsize = neuron->weights_size;
            w = 0;
#ifdef USE_AVX
            AVXMultiplyValue(wsize, previousLayer->avx_activation_cache, d,
                             gradient->weights, w, 0, 0, 0);
#endif
            for (; w < wsize; w++) {
                double prev_a = previousLayer->neurons[w]->activation;
                gradient->weights[w] = d * prev_a;
            }
        }
    }
    for (i = previousLayer->index; i > 0; i--) {
        PSLayer * layer = network->layers[i];
        previousLayer = network->layers[i - 1];
        nextLayer = network->layers[i + 1];
        lgradients = gradients[i - 1];
        int lsize = layer->size;
        PSLayerType ltype = layer->type;
        PSLayerType prev_ltype = previousLayer->type;
        if (FullyConnected == ltype) {
            delta = layer->delta;
            for (j = 0; j < lsize; j++) {
                PSNeuron * neuron = layer->neurons[j];
                double d = getDeltaForNeuron(neuron, layer,
                                             nextLayer, last_delta);
                delta[j] = d;
                PSGradient * gradient = &(lgradients[j]);
                gradient->bias = delta[j];
                w = 0;
                int wsize = neuron->weights_size;
#ifdef USE_AVX
                AVXMultiplyValue(wsize, previousLayer->avx_activation_cache, d,
                                 gradient->weights, w, 0, 0, 0);
#endif
                for (; w < wsize; w++) {
                    double prev_a = previousLayer->neurons[w]->activation;
                    gradient->weights[w] = d * prev_a;
                }
            }
        } else if (Pooling == ltype && Convolutional == prev_ltype) {
            delta = layer->delta;
            PSGetDeltaFunction _getDelta = NULL;
            if (nextLayer->type == Convolutional)
                _getDelta = getDeltaForConvolutionalNeuron;
            else
                _getDelta = getDeltaForNeuron;
            for (j = 0; j < lsize; j++) {
                PSNeuron * neuron = layer->neurons[j];
                delta[j] = _getDelta(neuron, layer, nextLayer, last_delta);
            }
            last_delta = delta;
            PSPoolingBackprop(layer, previousLayer, last_delta);
        } else if (Convolutional == ltype) {
            PSConvolutionalBackprop(layer, previousLayer, lgradients);
        } else {
            fprintf(stderr, "Backprop from %s to %s not suported!\n",
                    PSGetLayerTypeLabel(layer),
                    PSGetLayerTypeLabel(previousLayer));
            PSDeleteGradients(gradients, network);
            return NULL;
        }
        if (last_delta != delta) last_delta = delta;
    }
    return gradients;
}

PSGradient ** backpropThroughTime(PSNeuralNetwork * network, double * x,
                                  double * y, int times)
{
    if (network == NULL) return NULL;
    PSGradient ** gradients = createGradients(network);
    if (gradients == NULL) return NULL;
    int netsize = network->size;
    PSLayer * outputLayer = network->layers[netsize - 1];
    if (outputLayer->type != SoftMax) {
        PSErr("backpropThroughTime",
              "Recurrent networks require a Softmax output layer, "
              "current one is of type %s.", PSGetLayerTypeLabel(outputLayer));
        PSDeleteGradients(gradients, network);
        return NULL;
    }
    int onehot = (outputLayer->flags & FLAG_ONEHOT);
    int osize = outputLayer->size;
    int bptt_truncate = BPTT_TRUNCATE;

    int i, o, w, j, k, t;
    int ok = feedforwardThroughTime(network, x, times);
    if (!ok) {
        PSDeleteGradients(gradients, network);
        return NULL;
    }
    
    int last_t = times - 1;
    double * delta;
    double * last_delta;
    for (t = last_t; t >= 0; t--) {
        int lowest_t = t - bptt_truncate;
        if (lowest_t < 0) lowest_t = 0;
        PSLayer * previousLayer = NULL;
        PSLayer * nextLayer = NULL;
        int ysize = (onehot ? 1 : osize);
        int time_offset = t * ysize;
        double * time_y = y + time_offset;
        
        PSGradient * lgradients =
        gradients[netsize - 2];// No grad.for inputs
        previousLayer = network->layers[outputLayer->index - 1];
        nextLayer = NULL;
        
        delta = outputLayer->delta;
        last_delta = delta;
        
        double softmax_sum = 0.0;
        int apply_derivative = shouldApplyDerivative(network);
        // Calculate output deltas, output layer must be Softmax
        for (o = 0; o < osize; o++) {
            PSNeuron * neuron = outputLayer->neurons[o];
            PSRecurrentCell * cell = GetRecurrentCell(neuron);
            double o_val = cell->states[t];
            double y_val;
            if (onehot)
                y_val = ((int) *(time_y) == o);
            else
                y_val = time_y[o];
            double d = 0.0;
            y_val = (y_val < 1 ? 0 : 1);
            d = -(y_val - o_val);
            if (apply_derivative) d *= o_val;
            softmax_sum += d;
            delta[o] = d;
        }
        // Update gradients for output layer
        for (o = 0; o < osize; o++) {
            PSNeuron * neuron = outputLayer->neurons[o];
            PSRecurrentCell * cell = GetRecurrentCell(neuron);
            double o_val = cell->states[t];
            if (apply_derivative) delta[o] -= (o_val * softmax_sum);
            double d = delta[o];
            PSGradient * gradient = &(lgradients[o]);
            gradient->bias = d;
            w = 0;
#ifdef USE_AVX
            AVXMultiplyValue(neuron->weights_size,
                             previousLayer->avx_activation_cache, d,
                             gradient->weights, w,
                             1, t, AVX_STORE_MODE_ADD);
#endif
            for (; w < neuron->weights_size; w++) {
                PSNeuron * prev_neuron = previousLayer->neurons[w];
                PSRecurrentCell * prev_cell = GetRecurrentCell(prev_neuron);
                double prev_a = prev_cell->states[t];
                gradient->weights[w] += (d * prev_a);
            }
        }
        
        // Cycle through other layers
        for (i = previousLayer->index; i > 0; i--) {
            PSLayer * layer = network->layers[i];
            previousLayer = network->layers[i - 1];
            nextLayer = network->layers[i + 1];
            lgradients = gradients[i - 1];
            int lsize = layer->size;
            PSLayerType ltype = layer->type;
            int is_recurrent = (Recurrent == ltype);
            int is_lstm = (LSTM == ltype);
            if (!is_recurrent && !is_lstm) continue;
            //PSLayerType prev_ltype = previousLayer->type;
            
            delta = layer->delta;
            // Calculate layer deltas
            for (j = 0; j < lsize; j++) {
                PSNeuron * neuron = layer->neurons[j];
                PSRecurrentCell * cell = GetRecurrentCell(neuron);
                double sum = 0;
                for (k = 0; k < nextLayer->size; k++) {
                    PSNeuron * nextNeuron = nextLayer->neurons[k];
                    double weight = nextNeuron->weights[j];
                    double d = last_delta[k];
                    sum += (d * weight);
                }
                double dv = sum * layer->derivative(cell->states[t]);
                if (!is_lstm)
                    delta[j] = dv;
                else
                    delta[j] += dv;
                
                if (!is_recurrent && !is_lstm) {
                    PSGradient * gradient = &(lgradients[i]);
                    gradient->bias += dv;
                    int wsize = neuron->weights_size;
                    if (previousLayer->flags & FLAG_ONEHOT) {
                        PSLayerParameters * params = previousLayer->parameters;
                        if (params == NULL) {
                            fprintf(stderr, "Layer %d params are NULL!\n",
                                    previousLayer->index);
                            return NULL;
                        }
                        int vector_size = (int) params->parameters[0];
                        assert(vector_size > 0);
                        PSNeuron * prev_n = previousLayer->neurons[0];
                        PSRecurrentCell * prev_c = GetRecurrentCell(prev_n);
                        double prev_a = prev_c->states[t];
                        assert(prev_a < vector_size);
                        w = (int) prev_a;
                        gradient->weights[w] += dv;
                    } else {
                        for (w = 0; w < wsize; w++) {
                            PSNeuron * prev_n = previousLayer->neurons[w];
                            PSRecurrentCell * prev_c = GetRecurrentCell(prev_n);
                            double prev_a = prev_c->states[t];
                            gradient->weights[w] += (dv * prev_a);
                        }
                    }
                }
            }
            int ok = 1;
            if (is_recurrent)
                ok = PSRecurrentBackprop(layer, previousLayer, lowest_t,
                                         lgradients, t);
            else if (is_lstm)
                ok = PSLSTMBackprop(layer, previousLayer, lgradients, t);
            if (!ok) return NULL;
            last_delta = layer->delta;
        }
    }
    return gradients;
}

double updateWeights(PSNeuralNetwork * network, double * training_data,
                     int batch_size, int elements_count,
                     PSTrainingOptions* opts, double rate, ...)
{
    double r = rate / (double) batch_size;
    int i, j, k, w, netsize = network->size, dsize = netsize - 1, times;
    int training_data_size = network->input_size;
    int label_data_size = network->output_size;
    PSGradient ** gradients = createGradients(network);
    if (gradients == NULL) {
        network->status = STATUS_ERROR;
        return -999.0;
    }
    char * func = "updateWeights";
    PSGradient ** bp_gradients = NULL;
    double ** series = NULL;
    int is_recurrent = network->flags & FLAG_RECURRENT;
    if (is_recurrent) {
        va_list args;
        va_start(args, rate);
        series = va_arg(args, double**);
        va_end(args);
        if (series == NULL) {
            PSErr(func, "Series is NULL");
            network->status = STATUS_ERROR;
            PSDeleteGradients(gradients, network);
            return -999.0;
        }
    }
    double * x;
    double * y;
    for (i = 0; i < batch_size; i++) {
        if (series == NULL) {
            int element_size = training_data_size + label_data_size;
            x = training_data;
            y = training_data + training_data_size;
            training_data += element_size;
            bp_gradients = backprop(network, x, y);
        } else {
            x = series[i];
            times = (int) *(x++);
            if (times == 0) {
                PSErr(func, "Series len must b > 0. (batch = %d)", i);
                PSDeleteGradients(gradients, network);
                return -999.0;
            }
            y = x + (times * training_data_size);
            bp_gradients = backpropThroughTime(network, x, y, times);
        }
        if (bp_gradients == NULL) {
            network->status = STATUS_ERROR;
            PSDeleteGradients(gradients, network);
            return -999.0;
        }
        for (j = 0; j < dsize; j++) {
            PSLayer * layer = network->layers[j + 1];
            PSGradient * lgradients_bp = bp_gradients[j];
            PSGradient * lgradients = gradients[j];
            if (lgradients == NULL) continue;
            int lsize = layer->size;
            int wsize = 0;
            if (layer->type == Convolutional) {
                PSLayerParameters * params = layer->parameters;
                lsize = (int) (params->parameters[PARAM_FEATURE_COUNT]);
                int rsize = (int) (params->parameters[PARAM_REGION_SIZE]);
                wsize = rsize * rsize;
            }
            for (k = 0; k < lsize; k++) {
                if (!wsize) {
                    PSNeuron * neuron = layer->neurons[k];
                    wsize = neuron->weights_size;
                    if (layer->type == LSTM) wsize += 4; // LSTM biases
                }
                PSGradient * gradient_bp = &(lgradients_bp[k]);
                PSGradient * gradient = &(lgradients[k]);
                gradient->bias += gradient_bp->bias;
                w = 0;
#ifdef USE_AVX
                AVXSum(wsize, gradient->weights, gradient_bp->weights,
                       gradient->weights, w, 0);
#endif
                for (; w < wsize; w++)
                    gradient->weights[w] += gradient_bp->weights[w];
            }
        }
        PSDeleteGradients(bp_gradients, network);
    }
    
    double l1 = 0.0, l2 = 0.0, l2_loss = 0.0;
    if (opts != NULL) {
        //if (opts->l1_decay != 0.0) l1 = opts->l1_decay / elements_count;
        if (opts->l2_decay != 0.0) {
            l2 = opts->l2_decay / elements_count;
            l2 = (1 - (rate * l2));
        }
        l1 = (1 - (rate * l1));
    }
    
    for (i = 0; i < dsize; i++) {
        PSGradient * lgradients = gradients[i];
        if (lgradients == NULL) continue;
        PSLayer * layer = network->layers[i + 1];
        PSLayerType ltype = layer->type;
        int l_size;
        PSSharedParams * shared = NULL;
        if (ltype == Convolutional) {
            PSLayerParameters * params = layer->parameters;
            l_size = (int) (params->parameters[PARAM_FEATURE_COUNT]);
            shared = getConvSharedParams(layer);
        } else l_size = layer->size;
        int is_lstm = ltype == LSTM;
        for (j = 0; j < l_size; j++) {
            PSGradient * g = &(lgradients[j]);
            if (shared == NULL) {
                PSNeuron * neuron = layer->neurons[j];
                neuron->bias = neuron->bias - r * g->bias;
                int wsize = neuron->weights_size;
                if (is_lstm) PSUpdateLSTMBiases(neuron, g, r);
                k = 0;
#ifdef USE_AVX
                if (l2 != 0.0) {
                    int kk = 0;
                    AVXMultiplyValues(wsize, neuron->weights, l2, g->weights, r,
                                      neuron->weights, k, 0, 0,
                                      AVX_STORE_MODE_NORM, AVX_STORE_MODE_SUB);
                    AVXDotSquare(wsize, g->weights, l2_loss, kk, 0, 0);
                    if (kk < k) { // AVX Step Length could differ
                        for (; kk < k; kk++) {
                            double grad_w = g->weights[kk];
                            l2_loss += (grad_w * grad_w);
                        }
                    }
                } else {
                    AVXMultiplyValue(wsize, g->weights, r, neuron->weights,
                                     k, 0, 0, AVX_STORE_MODE_SUB);
                }
#endif
                for (; k < wsize; k++) {
                    double grad_w = g->weights[k];
                    if (l2 != 0.0) {
                        neuron->weights[k] *= l2;
                        l2_loss += (grad_w * grad_w);
                    }
                    neuron->weights[k] -= (r * grad_w);
                }
            } else {
                shared->biases[j] -= (r * g->bias);
                double * weights = shared->weights[j];
                k = 0;
#ifdef USE_AVX
                AVXMultiplyValue(shared->weights_size, g->weights, r, weights,
                                 k, 0, 0, AVX_STORE_MODE_SUB);
#endif
                for (; k < shared->weights_size; k++)
                    weights[k] -= (r * g->weights[k]);
            }
        }
    }
    PSDeleteGradients(gradients, network);
    PSLayer * out = network->layers[netsize - 1];
    int onehot = out->flags & FLAG_ONEHOT;
    if (onehot) label_data_size = 1;
    if (is_recurrent) label_data_size *= times;
    double outputs[label_data_size];
    for (i = 0; i < label_data_size; i++) {
        if (!is_recurrent)
            outputs[i] = out->neurons[i]->activation;
        else {
            if (onehot) {
                int idx = (int) *(y + i);
                PSNeuron * n = out->neurons[idx];
                PSRecurrentCell * cell = GetRecurrentCell(n);
                outputs[i] = cell->states[i];
            } else fetchRecurrentOutputState(out, outputs, i, 0);
        }
    }
    if (l2 != 0.0) l2_loss = (0.5 * (opts->l2_decay / batch_size) * l2_loss);
    int onehot_s = (onehot ? out->size : 0);
    return network->loss(outputs, y, label_data_size, onehot_s) + l2_loss;
}

double gradientDescent(PSNeuralNetwork * network,
                       double * training_data,
                       int element_size,
                       int elements_count,
                       double learning_rate,
                       int batch_size,
                       PSTrainingOptions * options,
                       int epochs) {
    int batches_count = elements_count / batch_size;
    double ** series = NULL;
    int flags = 0;
    if (options != NULL) flags = options->flags;
    if (network->flags & FLAG_RECURRENT) {
        if (series == NULL) {
            PSLayer * out = network->layers[network->size - 1];
            int o_size = (out->flags & FLAG_ONEHOT ? 1 : network->output_size);
            series = getRecurrentSeries(training_data,
                                        elements_count,
                                        network->input_size,
                                        o_size);
            if (series == NULL) {
                network->status = STATUS_ERROR;
                return -999.00;
            }
        }
        if (!(flags & TRAINING_NO_SHUFFLE))
            shuffleSeries(series, elements_count);
    } else {
        if (!(flags & TRAINING_NO_SHUFFLE))
            shuffle(training_data, elements_count, element_size);
    }
    int offset = (element_size * batch_size), i;
    double err = 0.0;
    for (i = 0; i < batches_count; i++) {
        network->current_batch = i;
        printf("\rEpoch %d/%d: batch %d/%d", network->current_epoch + 1, epochs,
               i + 1, batches_count);
        fflush(stdout);
        err += updateWeights(network, training_data, batch_size, elements_count,
                             options, learning_rate, series);
        if (network->status == STATUS_ERROR) {
            if (series != NULL) free(series - (i * batches_count));
            return -999.00;
        }
        if (series == NULL) training_data += offset;
        else series += batch_size;
    }
    if (series != NULL) free(series - (batch_size * batches_count));
    return err / (double) batches_count;
}

float validate(PSNeuralNetwork * network, double * test_data, int data_size,
               int log) {
    int i, j;
    float accuracy = 0.0f;
    int correct_results = 0;
    float correct_amount = 0.0f;
    PSLayer * output_layer = network->layers[network->size - 1];
    int input_size = network->input_size;
    int output_size = network->output_size;
    int y_size = output_size;
    int onehot = output_layer->flags & FLAG_ONEHOT;
    int element_size = input_size + output_size;
    int elements_count;
    double ** series = NULL;
    if (network->flags & FLAG_RECURRENT) {
        // First training data number for Recurrent networks must indicate
        // the data elements count
        elements_count = (int) *(test_data++);
        data_size--;
        if (onehot) y_size = 1;
        series = getRecurrentSeries(test_data,
                                    elements_count,
                                    input_size,
                                    y_size);
        if (series == NULL) {
            network->status = STATUS_ERROR;
            return -999.0f;
        }
    } else elements_count = data_size / element_size;
    //double outputs[output_size];
    if (log) printf("Test data elements: %d\n", elements_count);
    time_t start_t, end_t;
    char timestr[80];
    struct tm * tminfo;
    time(&start_t);
    tminfo = localtime(&start_t);
    strftime(timestr, 80, "%H:%M:%S", tminfo);
    if (log) printf("Testing started at %s\n", timestr);
    for (i = 0; i < elements_count; i++) {
        if (log) printf("\rTesting %d/%d", i + 1, elements_count);
        fflush(stdout);
        double * inputs = NULL;
        double * expected = NULL;
        int times = 0;
        if (series == NULL) {
            // Not Recurrent
            inputs = test_data;
            test_data += input_size;
            expected = test_data;
            
            int ok = PSFeedforward(network, inputs);
            if (!ok) {
                network->status = STATUS_ERROR;
                fprintf(stderr,
                        "\nAn error occurred while validating, aborting!\n");
                return -999.0;
            }
            
            double max = 0.0;
            int omax = 0;
            int emax = 0;
            for (j = 0; j < output_size; j++) {
                PSNeuron * neuron = output_layer->neurons[j];
                if (neuron->activation > max) {
                    max = neuron->activation;
                    omax = j;
                }
            }
            if (!onehot)
                emax = arrayMaxIndex(expected, output_size);
            else
                emax = *(expected + (times - 1));
            if (omax == emax) correct_results++;
            test_data += output_size;
        } else {
            // Recurrent
            inputs = series[i];
            times = (int) (*inputs);
            if (times == 0) {
                network->status = STATUS_ERROR;
                fprintf(stderr,
                        "\nAn error occurred while validating, aborting!\n");
                return -999.0;
            }
            expected = inputs + 1 + (times * input_size);
            
            int ok = PSFeedforward(network, inputs);
            if (!ok) {
                network->status = STATUS_ERROR;
                fprintf(stderr,
                        "\nAn error occurred while validating, aborting!\n");
                return -999.0;
            }
            
            int label_data_size = y_size * times;
            int correct_states = 0;
            double outputs[label_data_size];
            for (j = 0; j < label_data_size; j++) {
                fetchRecurrentOutputState(output_layer, outputs, j, onehot);
                if (onehot && (outputs[j] == expected[j])) correct_states++;
                else if (!onehot && j > 0 && (j % y_size) == 0) {
                    int t = (j / y_size) - 1;
                    int omax = arrayMaxIndex(outputs + (t * y_size), y_size);
                    int emax = arrayMaxIndex(expected + (t * y_size), y_size);
                    if (emax == omax) correct_states++;
                }
            }
            correct_amount += (float) correct_states / (float) times;
        }
    }
    if (log) printf("\n");
    time(&end_t);
    if (log) printf("Completed in %ld sec.\n", end_t - start_t);
    if (series == NULL) {
        accuracy = (float) correct_results / (float) elements_count;
        if (log) printf("Accuracy (%d/%d): %.2f\n",
                        correct_results, elements_count,accuracy);
    } else {
        accuracy = correct_amount / (float) elements_count;
        free(series);
        if (log) printf("Accuracy: %.2f\n", accuracy);
    }
    return accuracy;
}

void PSTrain(PSNeuralNetwork * network,
             double * training_data,
             int data_size,
             int epochs,
             double learning_rate,
             int batch_size,
             PSTrainingOptions * options,
             double * test_data,
             int test_size) {
    int i, elements_count;
    int element_size = network->input_size + network->output_size;
    int valid = PSVerifyNetwork(network);
    if (!valid) {
        network->status = STATUS_ERROR;
        return;
    }
    if (network->flags & FLAG_RECURRENT) {
        // First training data number for Recurrent networks must indicate
        // the data elements count
        elements_count = (int) *(training_data++);
        data_size--;
    } else elements_count = data_size / element_size;
    const char * name = network->name != NULL ? network->name : "UNNAMED";
    if (PSGlobalFlags & FLAG_LOG_COLORS) printf(BOLD);
    printf("Training network \"%s\"\n", name);
    if (PSGlobalFlags & FLAG_LOG_COLORS) printf(RESET);
    printf("Training data elements: %d\n", elements_count);
    printf("Batch Size: %d\n", batch_size);
    printf("Learning Rate: %.2f\n", learning_rate);
    if (options != NULL) printf("L2 Decay: %.2f\n", options->l2_decay);
    network->status = STATUS_TRAINING;
    time_t start_t, end_t, epoch_t;
    char timestr[80];
    struct tm * tminfo;
    time(&start_t);
    tminfo = localtime(&start_t);
    strftime(timestr, 80, "%H:%M:%S", tminfo);
    if (PSGlobalFlags & FLAG_LOG_COLORS) printf(CYAN);
    printf("Training started at %s\n", timestr);
    if (PSGlobalFlags & FLAG_LOG_COLORS) printf(WHITE);
    epoch_t = start_t;
    time_t e_t = epoch_t;
    double prev_err = 0.0;
    float acc = -999.99f;
    int adjust_rate = 0;
    if (options != NULL) adjust_rate = (options->flags & TRAINING_ADJUST_RATE);
    for (i = 0; i < epochs; i++) {
        network->current_epoch = i;
        double err = gradientDescent(network, training_data, element_size,
                                     elements_count, learning_rate,
                                     batch_size, options, epochs);
        if (network->status == STATUS_ERROR) {
            fprintf(stderr, "\nAn error occurred while training, aborting!\n");
            return;
        }
        char accuracy_msg[255] = "";
        if (test_data != NULL) {
            int batches_count = elements_count / batch_size;
            printf("\rEpoch %d/%d: batch %d/%d, validating...",
                   network->current_epoch + 1,
                   epochs,
                   network->current_batch + 1,
                   batches_count);
            acc = validate(network, test_data, test_size, 0);
            printf("\rEpoch %d/%d: batch %d/%d",
                   network->current_epoch + 1,
                   epochs,
                   network->current_batch + 1,
                   batches_count);
            sprintf(accuracy_msg, ", acc = %.2f,", acc);
        }
        time(&epoch_t);
        time_t elapsed_t = epoch_t - e_t;
        e_t = epoch_t;
        if (i > 0 && err > prev_err && adjust_rate)
            learning_rate *= 0.5;
        if (network->onEpochTrained != NULL)
            network->onEpochTrained(network, i, err, prev_err,
                                    acc, &learning_rate);
        prev_err = err;
        printf(", loss = %.2lf%s (%ld sec.)\n", err, accuracy_msg, elapsed_t);
    }
    time(&end_t);
    if (PSGlobalFlags & FLAG_LOG_COLORS) printf(GREEN);
    printf("Completed in %ld sec.\n", end_t - start_t);
    if (PSGlobalFlags & FLAG_LOG_COLORS) printf(WHITE);
    network->status = STATUS_TRAINED;
}

float PSTest(PSNeuralNetwork * network, double * test_data, int data_size) {
    return validate(network, test_data, data_size, 1);
}

int PSVerifyNetwork(PSNeuralNetwork * network) {
    char * func = "PSVerifyNetwork";
    if (network == NULL) {
        PSErr(func, "Network is NULL");
        return 0;
    }
    int size = network->size, i;
    PSLayer * previous = NULL;
    int onehot_input = 0;
    for (i = 0; i < size; i++) {
        PSLayer * layer = network->layers[i];
        if (layer == NULL) {
            PSErr(func, "Layer[%d] is NULL", i);
            return 0;
        }
        int ltype = layer->type;
        if (i == 0) {
            if (ltype != FullyConnected) {
                PSErr(func, "Layer[%d] type must be '%s'",
                      i, PSGetLabelForType(FullyConnected));
                return 0;
            }
            if (layer->flags & FLAG_ONEHOT) {
                PSLayerParameters * params = layer->parameters;
                onehot_input = 1;
                if (params == NULL) {
                    PSErr(func,
                          "Layer[%d] uses a onehot vector index as input, "
                          "but it has no parameters", i);
                    return 0;
                }
                if (params->count < 1) {
                    PSErr(func,
                          "Layer[%d] uses a onehot vector index as input, "
                          "but parameters count is < 1", i);
                    return 0;
                }
            }
        }
        if (ltype == Convolutional) {
            if (onehot_input) {
                PSErr(func, "ONEHOT input Layer is not supported on "
                      "Convolutional netowrks");
                return 0;
            }
            if (network->flags & FLAG_RECURRENT) {
                PSErr(func, "Sorry, Convolutional layers aren't yet supported "
                      "on recurrent networks :(");
                return 0;
            }
        }
        if (ltype == Pooling && previous && previous->type != Convolutional) {
            PSErr(func, "Layer[%d] type is Pooling, "
                  "but previous type is not Convolutional", i);
            return 0;
        }
        if (ltype != Pooling && previous &&  previous->type == Convolutional) {
            PSErr(func, "Layer[%d] previous type is "
                  "Convolutional, but type is not Pooling", i);
            return 0;
        }
        if (layer->activate == sigmoid &&
            layer->derivative != sigmoid_derivative) {
            PSErr(func,
                  "Layer[%d] activate function is sigmoid, "
                  "but derivative function is not sigmoid_derivative", i);
            return 0;
        }
        if (layer->activate == relu && layer->derivative != relu_derivative) {
            PSErr(func,
                  "Layer[%d] activate function is relu, "
                  "but derivative function is not relu_derivative", i);
            return 0;
        }
        if (layer->activate == tanh && layer->derivative != tanh_derivative) {
            PSErr(func,
                  "Layer[%d] activate function is tanh, "
                  "but derivative function is not tanh_derivative", i);
            return 0;
        }
        previous = layer;
    }
    if (network->flags & FLAG_RECURRENT) {
        PSLayer * output = network->layers[size - 1];
        if (output->type != SoftMax) {
            PSErr(func,
                  "Recurrent networks require a Softmax output layer, "
                  "current one is of type %s.",
                  PSGetLabelForType(output->type));
            return 0;
        }
    }
    return 1;
}
