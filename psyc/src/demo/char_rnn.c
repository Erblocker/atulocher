#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include "../psyc.h"
#include "char_training_data.h"

#define EPOCHS 300
#define LEARNING_RATE 0.0025
//#define LEARNING_RATE 0.01
#define BATCHES 1

#define strEq(s1,s2) (strcmp(s1, s2) == 0)


void TrainCallback (void * _net, int epoch, double loss,
                    double previous_loss, float accuracy,
                    double * rate)
{
    //if ((epoch % 2) != 0) return;
    PSNeuralNetwork * network = (PSNeuralNetwork*) _net;
    int i;
    double inputs[256];
    srand ( time(NULL) - i);
    int p = (rand() % 10) / 10.0f;
    inputs[0] = 1.0;
    inputs[1] = (double)(rand() % INPUT_SIZE);
    printf("\nSample:\n%s", characters[(int) inputs[1]]);
    for (i = 0; i < 254; i++) {
        srand ( time(NULL) + i);
        float p = (rand() % 10) / 10.0f;
        int idx = PSClassify(network, inputs);
        if (p <= 0.25f) {
            PSLayer * out = network->layers[network->size - 1];
            int o = 0;
            double omax = 0.0;
            int oidx = 0;
            for (; o < out->size; o++) {
                if (o == idx) continue;
                double a = out->neurons[o]->activation;
                if (a > omax) {
                    omax = a;
                    oidx = o;
                }
            }
            idx = oidx;
        }
        if (idx >= INPUT_SIZE) {
            fprintf(stderr, "Index %d >= %d", idx, INPUT_SIZE);
            return;
        }
        printf("%s", characters[idx]);
        inputs[0] += 1.0;
        inputs[(int) inputs[0]] = (double) idx;
    }
    printf("\n");
}

int main(int argc, char**argv){
    PSNeuralNetwork * network = PSCreateNetwork("TEST CHAR RNN");
    network->onEpochTrained = TrainCallback;
    
    int epochs = EPOCHS;
    double learning_rate = LEARNING_RATE;
    double l2_decay = 0.0;
    PSLayerType type = LSTM;
    /*double * vdataset = validation_data;
    int vdlen = EVAL_DATALEN;
    double * tdataset = test_data;
    int tdlen = TEST_DATALEN;*/
    int pretest = 0 ;
    
    int i;
    for (i = 0; i < argc; i++) {
        char * arg = argv[i];
        int next_idx = i + 1;
        if (strEq("--use-rnn", arg)) type = Recurrent;
        /*if (strEq("--same-dataset", arg)) {
            vdataset = training_data;
            vdlen = TRAIN_DATALEN;
            tdataset = training_data;
            tdlen = TRAIN_DATALEN;
        }*/
        if (strEq("--pre-test", arg)) pretest = 1;
        if (next_idx < argc) {
            char * next = argv[next_idx];
            if (strEq("--epochs", arg) || strEq("-e", arg)) {
                epochs = atoi(next);
                if (!epochs) {
                    fputs("Invalid epochs!", stderr);
                    return 1;
                }
            }
            if (strEq("--learning-rate", arg) || strEq("-r", arg)) {
                learning_rate = (double) atof(next);
                if (learning_rate == 0.0) {
                    fputs("Invalid learing rate!", stderr);
                    return 1;
                }
            }
            if (strEq("--l2-decay", arg))
                l2_decay = (double) atof(next);
        }
    }
    //printf("CHAR: %s\n", characters[6]);return 0;
    network->flags |= FLAG_ONEHOT;
    
    PSAddLayer(network, FullyConnected, INPUT_SIZE, NULL);
    PSAddLayer(network, type, INPUT_SIZE / 2, NULL);
    PSAddLayer(network, SoftMax, INPUT_SIZE, NULL);

    network->layers[network->size - 1]->flags |= FLAG_ONEHOT;
    
    printf("Epochs: %d\n", epochs);
    printf("Rate: %f\n", learning_rate);
    
    if (pretest) {
        PSTest(network, training_data, TRAIN_DATALEN);
        TrainCallback (network, 0, 0.0,
                       0.0, 0.0,
                       NULL);
    }
    //epochs = 2;
    PSTrainingOptions options = {
        .flags = TRAINING_NO_SHUFFLE,
        .l2_decay = l2_decay
    };
    printf("L2 Decay: %.2f\n", (float) l2_decay);
    PSTrain(network, training_data, TRAIN_DATALEN, epochs, learning_rate,
            BATCHES, &options, training_data, TRAIN_DATALEN);
    
    PSTest(network, training_data, TRAIN_DATALEN);

    PSDeleteNetwork(network);
    return 0;
}
