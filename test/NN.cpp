#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <atulocher/NN.hpp>
using namespace atulocher::NN;
//使用神经网络进行异或运算，输入为2个0~32767之间的数，前15节点为第1个数二进制，后15节点为第2个数二进制，输出为结果的二进制
void TestXor(){
    int layer[] = { 30,48,15 };
    BPAnn *bp = CreateBPAnn(0.25, 0.9, layer, 3, SIGMOD);
    double input[30], output[15];
    int error = 0;
    for (int j = 0; j < 100; ++j){
        for (int i = 0; i < 400; ++i){
            unsigned x = rand() & 32767;
            unsigned y = rand() & 32767;
            ToBinary(x << 15 | y, 30, input);
            ToBinary(x^y, 15, output);
            Train(input, output, bp);
        }
        //printf("%02d%%\n", j + 1);
    }
    printf("\nfinish train\n");
    for (int i = 0; i < 20; ++i){
        unsigned x = rand() & 32767;
        unsigned y = rand() & 32767;
        ToBinary(x << 15 | y, 30, input);
        Predict(input, output, bp);
        unsigned result = FromBinary(output, 15);
        if(result != (x^y)){
            ++error;
            
        }
        printf("%u^%u=%u\tpredict=%u\n", x, y, x^y, result);
    }
    printf("error = %d\n", error);
    //SaveBPAnn(bp,"BP.ann");
    DestroyBPAnn(bp);
}
void TestAnd(){
    int layer[] = { 30,48,15 };
    BPAnn *bp = CreateBPAnn(0.25, 0.9, layer, 3, SIGMOD);
    double input[30], output[15];
    int error = 0;
    for (int j = 0; j < 100; ++j){
        for (int i = 0; i < 400; ++i){
            unsigned x = rand() & 32767;
            unsigned y = rand() & 32767;
            ToBinary(x << 15 | y, 30, input);
            ToBinary(x&y, 15, output);
            Train(input, output, bp);
        }
        //printf("%02d%%\n", j + 1);
    }
    printf("\nfinish train\n");
    for (int i = 0; i < 20; ++i){
        unsigned x = rand() & 32767;
        unsigned y = rand() & 32767;
        ToBinary(x << 15 | y, 30, input);
        Predict(input, output, bp);
        unsigned result = FromBinary(output, 15);
        if(result != (x&y)){
            ++error;
            
        }
        printf("%u&%u=%u\tpredict=%u\n", x, y, x&y, result);
    }
    printf("error = %d\n", error);
    //SaveBPAnn(bp,"BP.ann");
    DestroyBPAnn(bp);
}
int main(){
  TestXor();
  TestAnd();
  return 0;
}