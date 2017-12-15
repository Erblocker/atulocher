#include <atulocher/dmsg.hpp>
using namespace atulocher;
int main(){
  Dmsg_client cl("127.0.0.1",8008);
  Dmsg_client_base::node d;
  snprintf(d.data,4096,"hello ");
  cl.sendMsg(&d);
  cl.sendMsg(&d);
  sleep(5);
  cl.sendMsg(&d);
}
