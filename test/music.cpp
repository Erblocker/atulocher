#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <atulocher/NN.hpp>
#include <unistd.h>
using namespace atulocher::NN;
using namespace std;
void traSolv(RNN * net,const char * line,int l){
  double inp[15],out[150];
  istringstream iss(line);
  ToBinary(l, 15, inp);
  for(int i=0;i<150;i++){
    out[i]=0;
  }
  while(iss.good()){
    int j;
    iss >> j;
    if(j>=150) continue;
    if(j<=0)   continue;
    out[j]=1;
  }
  net->train(inp,out);
}
void tra(RNN * net,const char * path){
  printf("train...\n");
  FILE * fp=fopen(path,"r");
  if(!fp){
    printf("read %s failed\n",path);
    return;
  }
  char line[4096];
  int l=0;
  while(!feof(fp)){
    fgets(line,4096,fp);
    traSolv(net,line,l);
    l++;
  }
  fclose(fp);
}
void pre(RNN * net,const char * path,int length,int sleepTime){
  printf("compose...\n");
  FILE * fp=fopen(path,"r");
  if(!fp){
    printf("open %s failed\n",path);
    return;
  }
  for(int i=0;i<length;i++){
    double inp[15],out[150];
    ToBinary(i, 15, inp);
    
    net->predict(inp,out);
    
    for(int j=0;j<150;j++){
      if(out[j]>0.5d){
        fprintf(fp,"%d ",j);
      }
    }
    fprintf(fp,"\n");
    if(sleepTime>0){
      sleep(sleepTime);
    }
  }
  fclose(fp);
}
int main(int narg,const char * arg[]){
  int layer[] = { 15,80,200,200,150 };
  RNN * net;
  string path;
  if(narg==3){
    net=new RNN(0.25, 0.9, layer, 3);
    path="music.ann";
  }else
  if(narg>=4){
    net=new RNN(arg[3]);
    path=arg[3];
  }else{
    printf(
      "atulocher::musicMaker\n"
      "Usage: [mode] [music score file] [ANN file] (length) (speed)\n"
      "mode: \"train\" or \"compose\"\n"
      "music score file format:\n"
      "  note note note...\n"
      "  note note note...\n"
      "  note note note...\n"
      "  ...\n"
      "example:(YRSSF Theme song)\n"
      "  10 6 10 8  6\n"
      "  10 8 10 8  6\n"
      "  8  6 10 8  6\n"
      "  5  8 10 8  6\n"
      "  2  6 9  7  5\n"
      "  2  8 9  7  5\n"
      "  2  6 9  7  5\n"
      "  2  8 9  7  5\n"
      "by cgoxopx\n"
    );
    return 1;
  }
  string mode=arg[1];
  if(mode=="train"){
    tra(net,arg[2]);
  }else
  if(mode=="compose"){
    if(narg>=5){
      if(narg>=6)
        pre(net,arg[2],atoi(arg[4]),0);
      else
        pre(net,arg[2],atoi(arg[4]),atoi(arg[5]));
    }else
      pre(net,arg[2],400,0);
  }
  
  net->save(path.c_str());
  delete net;
  return 0;
}