#define ATU_AS_SERVER
#include <atulocher/rpc.hpp>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <memory.h>
#include <sstream>
#include <list>
using namespace RakNet;
using namespace atulocher;
using namespace std;
RPC_Server server(8091);
static void creategraph(RakNet::BitStream * res,
                        RakNet::BitStream * ret,
                        RakNet::Packet*
){
    
}

static void destroygraph(RakNet::BitStream * res,
                         RakNet::BitStream * ret,
                         RakNet::Packet*
){
    
}

static void find(RakNet::BitStream * res,
                 RakNet::BitStream * ret,
                 RakNet::Packet*
){
    
}

static void add(RakNet::BitStream * res,
                RakNet::BitStream * ret,
                RakNet::Packet*
){
    
}

static void del(RakNet::BitStream * res,
                RakNet::BitStream * ret,
                RakNet::Packet*
){
    
}
int main(){
  server.RegisterBlockingFunction("graph_add",add);
  server.RegisterBlockingFunction("graph_del",del);
  server.RegisterBlockingFunction("graph_find",find);
  server.RegisterBlockingFunction("graph_create",creategraph);
  server.RegisterBlockingFunction("graph_destroy",destroygraph);
  server.run();
}