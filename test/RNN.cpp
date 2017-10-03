#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <time.h>
#include <atulocher/NN.hpp>
using namespace atulocher::NN;
void train(RNN & net){
  unsigned x = rand() & 32767;
  unsigned y = rand() & 32767;
  double inputx[15],inputy[15], output[15];
  ToBinary(x,   15, inputx);
  ToBinary(y,   15, inputy);
  ToBinary(x+y, 15, output);
  double in[2],out[1];
  in[0] =0;
  in[1] =0;
  out[0]=0;
  net.train(in,out,true);
  for(int i=14;i>=0;i--){
    in[0] =inputx[i];
    in[1] =inputy[i];
    out[0]=output[i];
    net.train(in,out,false);
  }
}
void test(RNN & net){
  unsigned x = rand() & 32767;
  unsigned y = rand() & 32767;
  double inputx[15],inputy[15], output[15],pre[15];
  ToBinary(x,   15, inputx);
  ToBinary(y,   15, inputy);
  ToBinary(x+y, 15, output);
  double in[2],out[1];
  in[0] =0;
  in[1] =0;
  net.predict(in,out,true);
  for(int i=14;i>=0;i--){
  //for(int i=0;i<15;i++){
    in[0] =inputx[i];
    in[1] =inputy[i];
    net.predict(in,out,false);
    pre[i]=out[0];
  }
  int th=FromBinary(pre, 15);;
  printf("%d+%d=%d \t truth=%d\n",x,y,th,x+y);
}
int main(){
  int layer[] = { 2,6,300,1 };
  RNN net(0.25, 0.9, layer, 4);
  int i;
  for(i=0;i<20000;i++)train(net);
  for(i=0;i<20;i++)test(net);
  net.save("test.ann");
  return 0;
}
