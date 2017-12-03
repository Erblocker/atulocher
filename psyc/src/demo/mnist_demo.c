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
#include "../psyc.h"
#include "../mnist.h"

#define INPUT_SIZE (28 * 28)
#define EPOCHS 30

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage %s IMAGE_FILE LABELS_FILE [TEST_FILES...]\n", argv[0]);
        return 1;
    }
    
    double * training_data = NULL;
    double * test_data = NULL;
    int testlen = 0;
    int datalen = 0;
    int loaded = 0;
    PSNeuralNetwork * network = PSCreateNetwork("MNIST Demo");
    if (network == NULL) {
        fprintf(stderr, "Could not create network!\n");
        return 1;
    }
    PSAddLayer(network, FullyConnected, INPUT_SIZE, NULL);
    PSAddLayer(network, FullyConnected, 30, NULL);
    PSAddLayer(network, FullyConnected, 10, NULL);
    
    if (network->size < 1) {
        fprintf(stderr, "Could not add all layers!\n");
        PSDeleteNetwork(network);
        return 1;
    }
    
    if (strcmp("--load", argv[1]) == 0) {
        loaded = PSLoadNetwork(network, argv[2]);
        if (!loaded) {
            printf("Could not load pretrained network!\n");
            return 1;
        }
        if (network->size < 1) {
            fprintf(stderr, "Could not add all layers!\n");
            PSDeleteNetwork(network);
            return 1;
        }
    } else {
        datalen = loadMNISTData(TRAINING_DATA, argv[1], argv[2],
                                &training_data);
        if (datalen == 0 || training_data == NULL) {
            printf("Could not load training data!\n");
            return 1;
        }
    }
    if (argc >= 5) {
        testlen = loadMNISTData(TEST_DATA, argv[3], argv[4],
                                &test_data);
    };
    
    printf("Data len: %d\n", datalen);
    
    if (!loaded) PSTrain(network, training_data, datalen, EPOCHS, 3, 10, NULL,
                         NULL, 0);
    if (network->status == STATUS_ERROR) {
        PSDeleteNetwork(network);
        if (training_data != NULL) free(training_data);
        if (test_data != NULL) free(test_data);
        return 1;
    }
    
    if (testlen > 0 && test_data != NULL) {
        printf("Test Data len: %d\n", testlen);
        PSTest(network, test_data, testlen);
    }
    
    PSDeleteNetwork(network);
    if (training_data != NULL) free(training_data);
    if (test_data != NULL) free(test_data);
    return 0;
}
