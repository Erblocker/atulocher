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
                        RakNet::Packet*){//name:creategraph
}

static void destroygraph(RakNet::BitStream * res,
                         RakNet::BitStream * ret,
                         RakNet::Packet*){//name:destroygraph
  
}

static void find(RakNet::BitStream * res,
                 RakNet::BitStream * ret,
                 RakNet::Packet*){//name:find
  
}

static void add(RakNet::BitStream * res,
                RakNet::BitStream * ret,
                RakNet::Packet*){//name:add
  
}

static void del(RakNet::BitStream * res,
                RakNet::BitStream * ret,
                RakNet::Packet*){//name:del
  
}
int main(){
  server.RegisterBlockingFunction("add",add);
  server.RegisterBlockingFunction("del",del);
  server.RegisterBlockingFunction("find",find);
  server.RegisterBlockingFunction("creategraph",creategraph);
  server.RegisterBlockingFunction("destroygraph",destroygraph);
  server.run();
}
