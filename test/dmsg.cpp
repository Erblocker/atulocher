#include <atulocher/dmsg.hpp>
using namespace atulocher;
class Msg:public Dmsg_queue{
  virtual void onGetMessage(node * p,int fd){
    printf("%s\n",p->data);
  }
}msg;
int main(){
  msg.run(8008);
}