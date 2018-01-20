#define ATU_AS_SERVER
extern "C"{
  #include <psyc/psyc.h>
}
#include <atulocher/rpc.hpp>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <memory.h>
using namespace RakNet;
using namespace atulocher;
using namespace std;
RPC_Server server(8091);
typedef vector<double> Vector;

static void creategraph(RakNet::BitStream * res,
                        RakNet::BitStream * ret,
                        RakNet::Packet*){
}

static void destroygraph(RakNet::BitStream * res,
                         RakNet::BitStream * ret,
                         RakNet::Packet*){
  
}

static void find(RakNet::BitStream * res,
                 RakNet::BitStream * ret,
                 RakNet::Packet*){
  
}

static void add(RakNet::BitStream * res,
                RakNet::BitStream * ret,
                RakNet::Packet*){
  
}

static void del(RakNet::BitStream * res,
                RakNet::BitStream * ret,
                RakNet::Packet*){
  
}

static void getobj(RakNet::BitStream * res,
                   RakNet::BitStream * ret,
                   RakNet::Packet*){
  
}

int main(){
  server.RegisterBlockingFunction("mind_add",add);
  server.RegisterBlockingFunction("mind_del",del);
  server.RegisterBlockingFunction("mind_find",find);
  server.RegisterBlockingFunction("mind_create",creategraph);
  server.RegisterBlockingFunction("mind_destroy",destroygraph);
  server.RegisterBlockingFunction("mind_getobj",getobj);
  server.run();
}
