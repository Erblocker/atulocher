#include <atulocher/dmsg.hpp>
using namespace atulocher;
class Msg:public Dmsg_server{
  virtual void onGetMessage(node * p,int fd,int sessid){
    printf("%s %d\n",p->data,sessid);
    Dmsg_node r;
    snprintf(r.data,ATU_CHUNK_SIZE,"recv ");
    //sendMsg(&r,sessid);
    send(fd,r.data,ATU_CHUNK_SIZE,0);
  }
  virtual void onLogout(int id){
    printf("logout:%d\n",id);
  }
  virtual void onLogin(int id){
    printf("login:%d\n",id);
  }
}msg;
int main(){
  msg.run(8008);
}
