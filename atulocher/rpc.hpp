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
    struct Header{
      int32_t id,len;
    };
    class MsgServer:public Dmsg_server{
      private:
      struct CliInfo{
        char * buf;
        int len,ptr,id,fd;
        MsgServer * owner;
        void init(int l){
          buf=(char*)malloc(l);
          len=l;
          ptr=0;
        }
        void destroy(){
          if(buf)free(buf);
          len=0;
          ptr=0;
          buf=NULL;
        }
        bool finish(){
          if(ptr>=len)
            return true;
          else
            return false;
        }
        void append(char c){
          //if(finish())return;
          buf[ptr]=c;
          ptr++;
        }
        #define autocheck \
          if(finish()){\
            callfunc();\
            destroy();\
            return;\
          }
        void add(void * pk){
          if(owner==NULL)return;
          auto bf=(char*)pk;
          if(len==0){
            
            auto hd=(Header*)pk;
            id=hd->id;
            
            #ifdef DEBUG
            printf("begin id=%x\n",id);
            printf("init:%x\n",hd->len);
            
            printf("data:");
            for(int i=0;i<ATU_CHUNK_SIZE;i++)
              printf("%c",bf[i]);
            printf("\n");
            #endif
            
            auto it=owner->funcs.find(id);
            if(it== owner->funcs.end())return;
            
            int tmplen;
            if(it->second->maxlen < hd->len)
              tmplen=it->second->maxlen;
            else
              tmplen=hd->len;
            
            this->init(tmplen);
            return;
          }
          
          for(int i=0;i<ATU_CHUNK_SIZE;i++){
            autocheck;
            append(bf[i]);
          }
          autocheck;
        }
        #undef autocheck
        void callfunc(){
          if(owner==NULL)return;
          auto it=owner->funcs.find(id);
          if(it== owner->funcs.end())return;
          it->second->func(buf,len,fd);
        }
      };
      public:
      class callback{
        public:
        int maxlen;
        virtual void func(void*,int,int)=0;
      };
      private:
      map<int,CliInfo> clis;
      map<int,callback*> funcs;
      public:
      void add(int id,callback * cb){
        funcs[id]=cb;
      }
      private:
      virtual void onGetMessage(node * p,int fd,int sessid){
        #ifdef DEBUG
        printf("onmsg:%d\n",sessid);
        #endif
        begin:
        auto it=clis.find(sessid);
        if(it==clis.end()){
          initcli(sessid);
          goto begin;
        }
        it->second.add(p->data);
      }
      virtual void onLogin(int id){
        #ifdef DEBUG
        printf("login:%d\n",id);
        #endif
        initcli(id);
      }
      virtual void initcli(int id){
        CliInfo & p=clis[id];
        p.fd=getfdbysid(id);
        p.id=id;
        p.len=0;
        p.ptr=0;
        p.buf=NULL;
        p.owner=this;
      }
      virtual void onLogout(int id){
        #ifdef DEBUG
        printf("logout:%d\n",id);
        #endif
        auto it=clis.find(id);
        if(it==clis.end())return;
        it->second.destroy();
        clis.erase(it);
      }
    };
    class MsgClient:public Dmsg_client{
      public:
      MsgClient(const char * addr,u_short port):Dmsg_client(addr,port){}
      virtual void call(int id,void * buf,int len){
        int ptr=0;
        auto cbuf=(char*)buf;
        node nbuf;
        nbuf.len=ATU_CHUNK_SIZE;
        
        auto hd=(Header*)nbuf.data;
        bzero(nbuf.data,ATU_CHUNK_SIZE);
        hd->id=id;
        hd->len=len;
        sendMsg(&nbuf);
        
        while(1){
          if(ptr>=len)break;
          bzero(nbuf.data,ATU_CHUNK_SIZE);
          for(int i=0;i<ATU_CHUNK_SIZE;i++){
            if(ptr>=len)break;
            nbuf.data[i]=cbuf[ptr];
            ++ptr;
          }
          sendMsg(&nbuf);
          
        }
      }
    };
  }
}
#endif
