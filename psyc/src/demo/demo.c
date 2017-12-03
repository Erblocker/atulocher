#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "psyc.h"

#define INPUTS_SIZE (28 * 28)


double normalized_rand() {
    srand(time(NULL));
    int r = rand();
    return ((double) r / (double) RAND_MAX);
}

int main(int argc, char** argv) {
    PSNeuralNetwork * network = PSCreateNetwork(NULL);
    PSAddLayer(network, FullyConnected, INPUTS_SIZE, NULL);
    PSAddLayer(network, FullyConnected, 30, NULL);
    PSAddLayer(network, FullyConnected, 10, NULL);
    
    double values[INPUTS_SIZE];
    int i;
    for (i = 0; i < INPUTS_SIZE; i++) {
        values[i] = normalized_rand();
    }
    PSFeedforward(network, values);
    
    PSDeleteNetwork(network);
    
    double nums[] = {1,2,3,4,5,6,7,8,9,10,11,12};
    testShuffle(nums, 6, 2);
    exit(0);
}
