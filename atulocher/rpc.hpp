#ifndef atulocher_rpc
#define atulocher_rpc
#include "dmsg.hpp"
#include <string>
namespace atulocher{
  namespace rpc{
    using namespace std;
    struct nint32{
      int32_t value;
      nint32(){
        value=0;
      }
      nint32(int32_t v){
        value=htonl(v);
      }
      nint32 & operator=(int32_t v){
        value=htonl(v);
        return *this;
      }
      nint32(const nint32 & v){
        value=v.value;
      }
      nint32 & operator=(const nint32 & v){
        value=v.value;
        return *this;
      }
      int32_t val(){
        return ntohl(value);
      }
      int32_t operator()(){
        return val();
      }
    };
    struct package{
      nint32 id;
      nint32 func;
      char   mode;
      char   buf[4000];
    };
    class base{
      public:
      inline void sendChunk(package * pk,int fd){
        Dmsg_node n;
        bzero(n.data,4096);
        memcpy(n.data,pk,sizeof(package));
        send(fd,n.data,4096,0);
      }
      void sendStr(const char * str,int fd){
        auto p=str;
        int i=0;
        package pk;
        while(*p){
          pk.buf[i]=*p;
          p++;
          i++;
          if(i==3999){
            pk.buf[3999]='\0';
            sendChunk(&pk,fd);
            i=0;
            bzero(&pk,sizeof(pk));
          }
        }
        pk.buf[3999]='\0';
        pk.buf[i]   ='\0';
        sendChunk(&pk,fd);
      }
    };
    class Client:public base,public Dmsg_client{
      public:
      virtual void call(int fid,const string & in,string & out){
        
      }
    };
    class Server:public Dmsg_server,public base{
      private:
      virtual void onGetMessage(node * p,int fd,int sessid){
        auto pk=(package*)p->data;
        
        int id=pk->id.val();
        int func=pk->func.val();
        
        switch(pk->mode){
          case 'd':
            sesses[sessid].erase(id);
          break;
          case 'a':
            pk->buf[3999]='\0';
            this->append(sessid,id,pk->buf,func);
          break;
          case 'c':
            this->call(sessid,id,fd);
          break;
        }
      }
      void onLogout(int sid){
        sesses.erase(sid);
      }
      typedef void(*callback)(const string &,string &);
      typedef pair<int,string>    status;
      typedef map<int,status>     staset;
      map<int,staset>             sesses;
      map<int,callback>           funcs;
      void setfunc(int sid,int id,int fid){
        sesses[sid][id].first=fid;
      }
      void append(int sid,int id,char * p,int fid){
        status & it=sesses[sid][id];
        it.first=fid;
        it.second+=p;
      }
      inline void call(int sid,int id,int fd){
        string res;
        call(sid,id,res);
        sendStr(res.c_str(),fd);
      }
      void call(int sid,int id,string & res){
        auto pp=sesses.find(sid);
        if(pp==sesses.end())return;
        
        staset & sp=pp->second;
        
        auto np=sp.find(id);
        if(np==sp.end())return;
          
        status & ps=np->second;
        auto fp=funcs.find(ps.first);
        if(fp==funcs.end())return;
        
        fp->second(ps.second,res);
      }
    };
  }
}
#endif