#ifndef atulocher_dmsg
#define atulocher_dmsg
#include <map>
#include <set>
#include <list>
#include <atomic>
#include <sys/epoll.h>
#include <memory.h>
#include <string>
#include <stdio.h>
#include <ctype.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <sys/file.h>
#include <ctype.h>
#include <strings.h>
#include <string.h>
#include <sys/stat.h>
#include <pthread.h>
#include <sys/wait.h>
#include <stdlib.h>
#include <mutex>
#include "threadpool.hpp"
#include "mempool.hpp"
#include "rwmutex.hpp"
#include "cyqueue.hpp"
namespace atulocher{
  class Dmsg_base{
    public:
    static void setnonblocking(int sock){
      int opts;
      opts = fcntl(sock, F_GETFL);
      if(opts < 0) {
        perror("fcntl(sock, GETFL)");
        return;
      }
      opts = opts | O_NONBLOCK;
      if(fcntl(sock, F_SETFL, opts) < 0) {
        perror("fcntl(sock, SETFL, opts)");
        return;
      }
    }
    static int startup(u_short *port){
      int rpcd = 0;
      struct sockaddr_in name;
      rpcd = socket(PF_INET, SOCK_STREAM, 0);
      if (rpcd == -1)
        return -1;
      int reuse = 1;
      if (setsockopt(rpcd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0) {
        return -1;
      }
      memset(&name, 0, sizeof(name));
      name.sin_family = AF_INET;
      name.sin_port = htons(*port);
      name.sin_addr.s_addr = htonl(INADDR_ANY);
      if (bind(rpcd, (struct sockaddr *)&name, sizeof(name)) < 0)
         return -1;
      if (*port == 0){
        int namelen = sizeof(name);
        if (getsockname(rpcd, (struct sockaddr *)&name, (socklen_t *)&namelen) == -1)
          return -1;
        *port = ntohs(name.sin_port);
      }
      if (listen(rpcd, 5) < 0)
        return -1;
      return (rpcd);
    }
    std::atomic<bool> running;
    int epfd;
    inline void stop(){
      running=false;
    }
    void run(u_short p,int maxnum=256){
      running=true;
      int listenfd = -1;
      u_short port = p;
      int connfd = -1;

      struct sockaddr_in client;
      int client_len = sizeof(client);
      
      signal(SIGPIPE,[](int){});
      //绑定监听端口
      listenfd = startup(&port);
      
      
      struct epoll_event ev, events[20];
      epfd = epoll_create(maxnum);
      setnonblocking(listenfd);
      ev.data.fd = listenfd;
      ev.events = EPOLLIN;
      epoll_ctl(epfd, EPOLL_CTL_ADD, listenfd, &ev);
      
      
      while (running){
        
        int nfds = epoll_wait(epfd, events, 20, 4000);
        loop();
        for(int i = 0; i < nfds; ++i) {
          if(events[i].data.fd == listenfd) {
            
            connfd = accept(listenfd, (struct sockaddr *)&client, (socklen_t *)&client_len);
            if(connfd < 0) {
              continue;
            }
            
            //printf("conn\n");
            onConnect(connfd);
            
            setnonblocking(connfd);
            ev.data.fd = connfd;
            ev.events = EPOLLIN | EPOLLHUP;
            epoll_ctl(epfd, EPOLL_CTL_ADD, connfd, &ev);
            
          }else if(events[i].events & EPOLLIN) {
            if((connfd = events[i].data.fd) < 0) continue;
            //处理请求
            char cbuf[4096];
            int len;
            if((len=read(connfd, cbuf, 4096)) <= 0) {
              
              onQuit(connfd);
              ev.data.fd = connfd;
              ev.events = 0;
              epoll_ctl(epfd, EPOLL_CTL_DEL, connfd, &ev);
              
            }else{
              
              //printf("msg\n");
              onMessage(connfd,cbuf,len);
              
              ev.data.fd = connfd;
              ev.events = EPOLLIN | EPOLLHUP;
              epoll_ctl(epfd, EPOLL_CTL_MOD, connfd, &ev);
              
            }
          }else if(events[i].events & EPOLLHUP) {
            if((connfd = events[i].data.fd) < 0) continue;
            
            //printf("hup\n");
            onQuit(connfd);
            
          }else if(events[i].events & EPOLLOUT) {
            if((connfd = events[i].data.fd) < 0) continue;
            onWriAble(connfd);
          }
        }
      }
      close(listenfd);
      close(epfd);
      destruct();
      return;
    }
    virtual void onMessage(int,char*,int){}
    virtual void onConnect(int){}
    virtual void onQuit(int){}
    virtual void onWriAble(int){}
    virtual void destruct(){}
    virtual void loop(){}
  };
  class Dmsg_client_base{
    public:
    struct node{
      node * next;
      char data[4096];
      unsigned int  len;
    };
    int sendMsg(int fd,node * d){
      return send(fd,d->data,4096,0);
    }
  };
  typedef Dmsg_client_base::node Dmsg_node;
  class Dmsg_client:public Dmsg_client_base{
    public:
    int sockfd;
    void init(sockaddr_in address){
      sockfd = socket(AF_INET, SOCK_STREAM, 0);
      connect(sockfd, (struct sockaddr *)&address, sizeof(address));
    }
    Dmsg_client(const char * addr,u_short port){
      struct sockaddr_in address;
      address.sin_family = AF_INET;
      address.sin_addr.s_addr = inet_addr(addr);
      address.sin_port = htons(port);
      init(address);
    }
    ~Dmsg_client(){
      if(sockfd!=-1)close(sockfd);
    }
    inline int sendMsg(node * d){
      return send(sockfd,d->data,4096,0);
    }
    inline int recvMsg(node * d){
      return read(sockfd,d->data,4096);
    }
  };
  class Dmsg_server:public Dmsg_base,public Dmsg_client_base{
    public:
    Dmsg_server(int cyl=256):cy(cyl){}
    private:
    cyqueue<std::pair<node*,int> > cy;//环形缓冲区
    struct conn{
      conn * next;
      node * data;
      int    sessionid;
      std::list<node*>waitforsend;
    };
    std::atomic<int> session;
    std::map<int,conn*> conns;
    std::map<int,int>   sessions;
    struct status{
      status * next;
      int fd;
      node * data;
      Dmsg_server * self;
      int sessionid;
    };
    mempool<status>    spool;
    mempool<conn>      cpool;
    mempool<node>      npool;
    static void * accreq(void * arg){
      auto self=(status*)arg;
      self->self->onGetMessage(self->data,self->fd,self->sessionid);
      self->self->npool.del(self->data);
      self->self->spool.del(self);
    }
    inline void resetConn(int connfd){
      auto it=conns.find(connfd);
      conn * cp;
      if(it==conns.end()){
        cp=cpool.get();
        conns[connfd]=cp;
      }else{
        cp=it->second;
      }
      cp->sessionid=session;
      sessions[cp->sessionid]=connfd;
      onLogin(cp->sessionid);
      session++;
      for(auto nnp:cp->waitforsend)
        npool.del(nnp);
      cp->waitforsend.clear();
      cleanConn(cp);
    }
    inline void cleanConn(conn * cp){
      if(cp->data==NULL)cp->data=npool.get();
      auto p=cp->data;
      p->len=0;
    }
    inline void removeConn(conn * cp){
      if(cp->data){
        npool.del(cp->data);
      }
      cp->waitforsend.clear();
      cpool.del(cp);
    }
    inline void charAppend(conn * cp,char ch,int connfd){
      auto p=cp->data;
      checkbuf(cp,connfd);
      p->data[p->len]=ch;
      p->len++;
      return;
    }
    inline void checkbuf(conn * cp,int connfd){
      auto p=cp->data;
      if(p->len>=4096){
        cp->data=npool.get();
        cp->data->len=0;
        auto sp=spool.get();
        sp->self=this;
        sp->fd=connfd;
        sp->data=p;
        sp->sessionid=cp->sessionid;
        threadpool::add(accreq,sp);
      }
    }
    inline bool connAppend(conn * cp,int connfd){
      auto p=cp->data;
      checkbuf(cp,connfd);
      char buf;
      if(read(connfd,&buf,1)<=0)return true;
      p->data[p->len]=buf;
      p->len++;
      return false;
    }
    virtual void onMessage(int connfd,char * fst,int len){
      auto it=conns.find(connfd);
      conn * cp;
      if(it==conns.end())return;
      cp=it->second;
      
      if(cp->data==NULL)
        cleanConn(cp);
      else
        checkbuf(cp,connfd);
      
      for(int j=0;j<len;j++)
        charAppend(cp,fst[j],connfd);
      
      checkbuf(cp,connfd);
      
      while(1){
        char buf[4096];
        int slen=read(connfd,buf,4096);
        if(slen<=0)return;
        for(int j=0;j<slen;j++)
          charAppend(cp,buf[j],connfd);
        checkbuf(cp,connfd);
        if(slen<4096)return;
      }
      
    }
    virtual void onConnect(int connfd){
      resetConn(connfd);
    }
    virtual void onQuit(int connfd){
      auto it=conns.find(connfd);
      if(it==conns.end())return;
      
      auto sit=sessions.find(it->second->sessionid);
      if(sit!=sessions.end()){
        
        onLogout(it->second->sessionid);
        sessions.erase(sit);
      }
      
      removeConn(it->second);
      conns.erase(it);
    }
    virtual void destruct(){
      for(auto it:conns){
        removeConn(it.second);
      }
    }
    virtual void onWriAble(int connfd){
      auto it=conns.find(connfd);
      conn * cp;
      if(it==conns.end())return;
      cp=it->second;
      for(auto it:cp->waitforsend){
        send(connfd,it->data,4096,0);
        npool.del(it);
      }
      cp->waitforsend.clear();
      
      epoll_event ev;
      ev.data.fd = connfd;
      ev.events = EPOLLIN | EPOLLHUP;
      epoll_ctl(epfd, EPOLL_CTL_MOD, connfd, &ev);
    }
    virtual void loop(){
      std::pair<node*,int> buf;
      while(cy.pop_noblock(&buf)){
      
        auto sit=sessions.find(buf.second);
        if(sit==sessions.end())continue;
        
        int connfd=sit->second;
        auto it=conns.find(connfd);
        conn * cp;
        if(it==conns.end())continue;
        cp=it->second;
        node * bd=buf.first;
        bd->len=4096;
        cp->waitforsend.push_back(bd);
      
        epoll_event ev;
        ev.data.fd = connfd;
        ev.events = EPOLLIN | EPOLLHUP | EPOLLOUT;
        epoll_ctl(epfd, EPOLL_CTL_MOD, connfd, &ev);
      }
    }
    public:
    virtual void sendMsg(node * d,int sessid){
      node * bd=npool.get();
      memcpy(bd->data,(const char*)d->data,4096);
      bd->len=4096;
      std::pair<node*,int> buf;
      buf.first=bd;
      buf.second=sessid;
      cy.push(buf);
    }
    inline void delnode(node * p){
      npool.del(p);
    }
    virtual void onGetMessage(node * p,int fd,int sessionid)=0;
    virtual void onLogin(int){}
    virtual void onLogout(int){}
  };
}
#endif