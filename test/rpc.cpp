#include <atulocher/rpc.hpp>
using namespace atulocher::rpc;
class cb:public MsgServer::callback{
  public:
  cb(){
    maxlen=100;
  }
  virtual void func(void * buf,int,int){
    printf("%s\n",buf);
  }
}cbv;
int main(){
  MsgServer sv;
  sv.add(1,&cbv);
  sv.run(8008);
}
