#include <atulocher/dmsg.hpp>
using namespace atulocher;
int main(){
  Dmsg_client cl("127.0.0.1",8008);
  Dmsg_client_base::node d;
  for(int i=0;i<5;i++){
    snprintf(d.data,ATU_CHUNK_SIZE,"hello ");
    cl.sendMsg(&d);
    sleep(1);
    cl.recvMsg(&d);
    printf("%s\n",d.data);
  }
}
