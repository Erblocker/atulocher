#ifndef tinyRPCd_main
#define tinyRPCd_main
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
namespace tinyRPCd{
  class status{
    public:
    typedef enum{
      REQUEST=0x01,
      RESPONSE=0x02,
      ERROR=0x03
    }Method;
    typedef struct{
      Method method;
      char   name[32];
      int    length;
    }Header;
    Header header;
    char * buf;
    int buflen;
    int fd;
    void destroy(){
      close(fd);
    }
    void init(int fd){
      this->fd=fd;
      bzero(&header,sizeof(Header));
      read(fd,&header,sizeof(Header));
      int hr=read(fd,buf,buflen);
      if(hr==-1)return;
      if(hr<header.length){
        int leave=header.length-hr;
        char rubbish;
        for(int i=0;i<leave;i++)
          read(fd,&rubbish,1);
      }
    }
    void ret(void * data,int len){
      header.length=len;
      send(fd,&header,sizeof(Header),0);
      send(fd,data,len,0);
    }
  };
  class server{
    
  };
  class client{
  
  };
}
#endif