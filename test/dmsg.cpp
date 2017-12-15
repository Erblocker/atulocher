#include <atulocher/dmsg.hpp>
using namespace atulocher;
class Msg:public Dmsg_server{
  virtual void onGetMessage(node * p,int fd,int sessid){
    printf("%s %d\n",p->data,sessid);
  }
}msg;
int main(){
  msg.run(8008);
}
