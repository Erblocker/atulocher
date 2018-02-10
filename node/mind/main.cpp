#define ATU_AS_SERVER
extern "C"{
  #include <psyc/psyc.h>
}
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
typedef vector<double> Vector;
unordered_map <string,Vector> objs;

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

void getobj(const list<string> & bufw,Vector & bufv,int len){
  
}

static void getobj(RakNet::BitStream * res,
                   RakNet::BitStream * ret,
                   RakNet::Packet*){
  int i;
  RakNet::RakString data;
  int offset=res->GetReadOffset();
  if(!(ret->ReadCompressed(data)))return;
  
  istringstream iss(data.C_String());
  
  int fd,len;
  iss>>fd;
  iss>>len;
  
  Vector bufv(len);
  list<string> bufw;
  string bufs;
  
  while(1){
    bufs.clear();
    iss>>bufs;
    if(bufs.empty())
      break;
    else
      bufw.push_back(bufs);
  }
  getobj(bufw,bufv,len);
  for(i=0;i<len;i++)
    ret->Write(bufv[i]);
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
