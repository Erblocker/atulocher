#define ATU_AS_SERVER
extern "C"{
  #include <psyc/psyc.h>
}
#include <atulocher/rpc.hpp>
#include <unordered_map>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <memory.h>
using namespace RakNet;
using namespace atulocher;
using namespace std;
RPC_Server server(8090);
map<int,PSNeuralNetwork*> networks;
static void predict(RakNet::BitStream * res,
                    RakNet::BitStream * ret,
                    RakNet::Packet*){//name:PSPredict
  double buf;
  int fd,data_size,tm,maxlen;
  int i;
  res->Read(buf); data_size =(int)buf;
  res->Read(buf); fd        =(int)buf;
  res->Read(buf); tm        =(int)buf;
  res->Read(buf); maxlen    =(int)buf;
  
  auto it=networks.find(fd);
  if(it==networks.end())return;
  PSNeuralNetwork * network=it->second;
  
  double * input_data  =(double*)malloc(sizeof(double)*data_size);
  double * output_data =(double*)malloc(sizeof(double)*data_size);
  
  for(i=0;i<data_size;i++) res->Read(input_data[i]);
  
  PSPredict(network,input_data,output_data,data_size,tm,maxlen);
  
  for(i=0;i<data_size;i++) ret->Write(output_data[i]);
  
  free(input_data);
  free(output_data);
}
int main(){
  server.RegisterBlockingFunction("PSPredict",predict);
  server.run();
}
