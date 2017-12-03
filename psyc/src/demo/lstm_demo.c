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
#include <stdlib.h>
#include <string.h>

#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>

#include <fenv.h>
#include <xmmintrin.h>

#include "../psyc.h"
#include "w2v_training_data.h"

#define BATCHES 1
#define EPOCHS 60
#define LEARNING_RATE 0.0025

void handler(int sig) {
    void *array[10];
    size_t size;
    
    // get void*'s for all entries on the stack
    size = backtrace(array, 10);
    
    // print out all the frames to stderr
    fprintf(stdout, "Error: signal %d:\n", sig);
    backtrace_symbols_fd(array, size, STDOUT_FILENO);
    exit(1);
}

int main(int argc, char** argv) {
    
    signal(SIGSEGV, handler);
    signal(8, handler);
    _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~_MM_MASK_INVALID);
    
    const char * pretrained_file = NULL;
    
    if (argc >= 3 && strcmp("--load", argv[1]) == 0) {
        pretrained_file = argv[2];
    }
    
    PSNeuralNetwork * network = PSCreateNetwork("RNN Demo");
    if (network == NULL) {
        fprintf(stderr, "Could not create network!\n");
        return 1;
    }
    network->flags |= FLAG_ONEHOT;
    
    if (pretrained_file == NULL) {
        PSAddLayer(network, FullyConnected, VOCABULARY_SIZE, NULL);
        PSAddLayer(network, LSTM, VOCABULARY_SIZE / 10, NULL);
        PSAddLayer(network, SoftMax, VOCABULARY_SIZE, NULL);
        network->layers[network->size - 1]->flags |= FLAG_ONEHOT;
        if (network->size < 1) {
            fprintf(stderr, "Could not add all layers!\n");
            PSDeleteNetwork(network);
            return 1;
        }
    } else {
        int loaded = PSLoadNetwork(network, pretrained_file);
        if (!loaded) {
            printf("Could not load pretrained data %s\n", pretrained_file);
            PSDeleteNetwork(network);
            return 1;
        }
        if (network->size < 1) {
            fprintf(stderr, "Could not add all layers!\n");
            PSDeleteNetwork(network);
            return 1;
        }
    }
    
    PSTrainingOptions options = {
        .flags = TRAINING_NO_SHUFFLE,
        .l2_decay = 0.0
    };
    PSTrain(network, training_data, TRAIN_DATA_LEN, EPOCHS, LEARNING_RATE,
            BATCHES, &options,
            training_data, TRAIN_DATA_LEN);
    
    if (TEST_DATA_LEN > 0) {
        printf("Test Data len: %d\n", TEST_DATA_LEN);
        PSTest(network, test_data, TEST_DATA_LEN);
    }
    if (pretrained_file == NULL)
        PSSaveNetwork(network, "/tmp/pretrained.lstm.data");
    PSDeleteNetwork(network);
    //free(training_data);
    //if (TEST_DATA_LEN) free(test_data);
    return 0;
}
