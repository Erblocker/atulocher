#ifndef atulocher_rpc
#define atulocher_rpc
#include <map>
#include <set>
#include <string>
#include <atomic>
#include "cppjieba/limonp/StdExtension.hpp"
#include <raknet/RPC4Plugin.h>
#include <raknet/RakPeerInterface.h>
#include <stdio.h>
#include <raknet/Kbhit.h>
#include <string.h>
#include <stdlib.h>
#include <raknet/RakSleep.h>
#include <raknet/BitStream.h>
#include <raknet/MessageIdentifiers.h>
#include <raknet/Gets.h>
namespace atulocher{
  class RPC_Server{
    public:
    RakNet::RPC4               rpc;
    RakNet::RakPeerInterface * rakPeer;
    RakNet::SocketDescriptor   sd1;
    std::atomic<bool> running;
    
    inline void RegisterSlot(const char * name,void(*callback)(RakNet::BitStream*,RakNet::Packet*)){
      rpc.RegisterSlot(name,callback,0);
    }
    inline void RegisterBlockingFunction(
        const char * name,
        void(*callback)(RakNet::BitStream*,RakNet::BitStream*,RakNet::Packet*)
    ){
      rpc.RegisterBlockingFunction(name,callback);
    }
    RPC_Server(int port):sd1(port,0),rpc(){
      rakPeer=RakNet::RakPeerInterface::GetInstance();
      rakPeer->Startup(8,&sd1,1);
      rakPeer->SetMaximumIncomingConnections(8);
      rakPeer->AllowConnectionResponseIPMigration(false);
      rakPeer->AttachPlugin(&rpc);
    }
    ~RPC_Server(){
      rakPeer->Shutdown(100,0);
      RakNet::RakPeerInterface::DestroyInstance(rakPeer);
    }
    
    virtual void stop(){
      running=false;
    }
    virtual void run(){
      running=true;
      RakNet::Packet *packet;
      while (running){
        RakSleep(100);
        for (packet=rakPeer->Receive(); packet; rakPeer->DeallocatePacket(packet), packet=rakPeer->Receive()){
          if(!running)return;
        }
      }
    }
  };
  class RPC{
    public:
    class remote{//server config
      public:
      RakNet::RPC4               rpc;
      RakNet::RakPeerInterface * rakPeer;
      RakNet::SocketDescriptor   sd1;
      
      remote(const char * ip,int port):rpc(),sd1(0,0){
       rakPeer=RakNet::RakPeerInterface::GetInstance();
       rakPeer->Startup(8,&sd1,1);
       rakPeer->SetMaximumIncomingConnections(8);
       rakPeer->AllowConnectionResponseIPMigration(false);
       rakPeer->AttachPlugin(&rpc);
       rakPeer->Connect(ip, port, 0, 0);
      }
      
      ~remote(){
        rakPeer->Shutdown(100,0);
        RakNet::RakPeerInterface::DestroyInstance(rakPeer);
      }
      
    };
    class function{
      public:
      remote * server;
      std::string name;
      bool blocking;
      void call(RakNet::BitStream *bitStream, RakNet::BitStream *returnData=NULL){
        if(blocking){
          if(bitStream ==NULL)return;
          if(returnData==NULL)return;
          server->rpc.CallBlocking(
            name.c_str(),
            bitStream,
            HIGH_PRIORITY,
            RELIABLE_ORDERED,
            0,
            server->rakPeer->GetSystemAddressFromIndex(0),
            returnData
          );
        }else{
          if(bitStream ==NULL)return;
          server->rpc.Signal(
            name.c_str(),
            bitStream,
            HIGH_PRIORITY,
            RELIABLE_ORDERED,
            0,
            server->rakPeer->GetSystemAddressFromIndex(0),
            false,
            true
          );
        }
      }
    };
    std::unordered_map<std::string,remote*> remotes;
    std::unordered_map<std::string,function> funcs;
    RPC(){
      for(auto it:remotes){
        delete it.second;
      }
    }
    void add(const std::string & sv,int pt,const std::string name,bool bk=true){
      char nms[128];
      snprintf(nms,128,"%s:%d",sv.c_str(),pt);
      auto it=remotes.find(nms);
      remote * rpt;
      
      if(it==remotes.end()){//add server list
        rpt=new remote(sv.c_str(),pt);
        remotes[nms]=rpt;
      }else{
        rpt=it->second;
      }
      
      //add func list
      function & fp=funcs[name];
      fp.server=rpt;
      fp.name=name;
      fp.blocking=bk;
    }
    void call(std::string name,RakNet::BitStream *bitStream, RakNet::BitStream *returnData=NULL){
      auto it=funcs.find(name);
      if(it==funcs.end())return;
      it->second.call(bitStream,returnData);
    }
  };
  #ifndef ATU_AS_SERVER
  RPC rpc;
  #endif
}
#endif
