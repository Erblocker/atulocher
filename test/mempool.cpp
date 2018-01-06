#include <atulocher/mempool.hpp>
class test{
  public:
  test * next,* gc_next;
};
int testpool(){
  atulocher::mempool<test> mp;
  for(int i=0;i<100;i++){
    auto p=mp.get();
    mp.del(p);
  }
}
int main(){
  for(int i=0;i<100;i++)testpool();
}
