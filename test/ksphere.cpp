#include <atulocher/word2vec.hpp>
using namespace atulocher;
using namespace std;
int main(){
  ksphere kn("test.sph");
  printf("inited\n");
  string a="1111";
  string b="2222";
  kn.addaxion(a,b);
  kn.negate("1111");
  ksphere::adder ar(&kn);
  ar.mean(a,0.5d);
  ar.add(std::string("test"),std::string("t"));
  
  auto posi=ar.position;
  printf("position:(%lf,%lf,%lf)\n",posi.x,posi.y,posi.z);
  
  kn.getnear(ar.position,[](ksphere::knowledge*p,void*){
    auto posi=p->obj.position;
    printf("position:(%lf,%lf,%lf)\n",posi.x,posi.y,posi.z);
  },1000,NULL);
  
  double bin[16];
  ksphere::vec2bin(posi,bin,16);
  //如果你需要使用二进制的数据，调用vec2bin就行了
  printf("bin:");
  for(int i=0;i<16;i++){
    printf("%d",(int)(bin[i]));
  }
  printf("\n");
  return 0;
}