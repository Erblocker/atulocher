#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <time.h>
//#include <atulocher/threadpool.hpp>
#include <atulocher/NN.hpp>
using namespace atulocher::NN;
void TestXor(){
    int layer[] = { 30,48,15 };
    NN nn(0.25, 0.9, layer, 3, SIGMOD);
    double input[30], output[15];
    int error = 0;
    for (int j = 0; j < 100; ++j){
        for (int i = 0; i < 400; ++i){
            unsigned x = rand() & 32767;
            unsigned y = rand() & 32767;
            ToBinary(x << 15 | y, 30, input);
            ToBinary(x^y, 15, output);
            nn.train(input, output);
        }
        //printf("%02d%%\n", j + 1);
    }
    printf("\nfinish train\n");
    for (int i = 0; i < 20; ++i){
        unsigned x = rand() & 32767;
        unsigned y = rand() & 32767;
        ToBinary(x << 15 | y, 30, input);
        nn.predict(input, output);
        unsigned result = FromBinary(output, 15);
        if(result != (x^y)){
            ++error;
            
        }
        printf("%u^%u=%u\tpredict=%u\n", x, y, x^y, result);
    }
    printf("error = %d\n", error);
}
void TestAnd(){
    int layer[] = { 30,48,15 };
    NN nn(0.25, 0.9, layer, 3, SIGMOD);
    double input[30], output[15];
    int error = 0;
    for (int j = 0; j < 100; ++j){
        for (int i = 0; i < 400; ++i){
            unsigned x = rand() & 32767;
            unsigned y = rand() & 32767;
            ToBinary(x << 15 | y, 30, input);
            ToBinary(x&y, 15, output);
            nn.train(input, output);
        }
        //printf("%02d%%\n", j + 1);
    }
    printf("\nfinish train\n");
    for (int i = 0; i < 20; ++i){
        unsigned x = rand() & 32767;
        unsigned y = rand() & 32767;
        ToBinary(x << 15 | y, 30, input);
        nn.predict(input, output);
        unsigned result = FromBinary(output, 15);
        if(result != (x&y)){
            ++error;
            
        }
        printf("%u&%u=%u\tpredict=%u\n", x, y, x&y, result);
    }
    printf("error = %d\n", error);
}
void TestAdd(){
    int layer[] = { 30,48,64,15 };//没办法啊，不弄那么深，学不会……
    NN nn(0.1, 0.1, layer, 4, SIGMOD);
    double input[30], output[15];
    int error = 0;
    for (int j = 0; j < 100; ++j){
        for (int i = 0; i < 400; ++i){
            unsigned x = rand() & 32767;
            unsigned y = rand() & 32767;
            ToBinary(x << 15 | y, 30, input);
            ToBinary(x+y, 15, output);
            nn.train(input, output);
        }
        //printf("%02d%%\n", j + 1);
    }
    printf("\nfinish train\n");
    for (int i = 0; i < 20; ++i){
        unsigned x = rand() & 32767;
        unsigned y = rand() & 32767;
        ToBinary(x << 15 | y, 30, input);
        nn.predict(input, output);
        unsigned result = FromBinary(output, 15);
        if(result != (x+y)){
            ++error;
            
        }
        printf("%u+%u=%u\tpredict=%u\n", x, y, x+y, result);
    }
    printf("error = %d\n", error);
}
int main(int i,const char ** arg){
  printf("\ntrain\n");
    
  if(i==2){
    std::string s(arg[1]);
    if(s=="add")
      TestAdd();
    else
    if(s=="xor")
      TestXor();
    else
    if(s=="and")
      TestAnd();
    
  }else{
    TestXor();
    TestAnd();
    TestAdd();
  }
  return 0;
}