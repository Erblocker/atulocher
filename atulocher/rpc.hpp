#ifndef atulocher_rpc
#define atulocher_rpc
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
namespace atulocher{
  class rpc_base{
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
  };
  class rpc_server{
  
  };
  class rpc_client{
  
  };
}
#endif