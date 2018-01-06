#include <atulocher/rpc.hpp>
using namespace atulocher::rpc;
int main(){
  MsgClient cl("127.0.0.1",8008);
  char buf[]="hello";
  cl.call(1,buf,sizeof(buf));
  sleep(1);
}
