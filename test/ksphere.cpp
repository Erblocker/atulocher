#include <atulocher/word2vec.hpp>
using namespace atulocher;
using namespace std;
int main(){
  ksphere kn("test.sph");
  printf("inited\n");
  string a1="this_is_a1";
  string a2="this_is_a2";
  string b="value_of_b";
  kn.addaxion(a1,b);
  kn.addaxion(a2,b);
  kn.negate(a1);
  
  ksphere::adder ar(&kn);
  ar.mean(a1,0.5d);
  ar.mean(a2,0.5d);
  ar.add(std::string("test"),std::string("t"));
  
  
  ksphere::adder ar2(&kn);
  ar2.mean(std::string("test"),0.5d);
  ar2.mean(std::string("test3"),0.5d,false);
  ar2.add(std::string("test2"),std::string("t"));
  
  for(auto it : kn.known["test"]->dep)
    printf("test_depend_axion:position:(%lf,%lf,%lf)\n",
      it.ptr->obj.position.x,
      it.ptr->obj.position.y,
      it.ptr->obj.position.z
    );
  
  printf("\n");
  
  for(auto it : kn.known["test2"]->dep)
    printf("test_depend_axion:position:(%lf,%lf,%lf)\n",
      it.ptr->obj.position.x,
      it.ptr->obj.position.y,
      it.ptr->obj.position.z
    );
  
  printf("\n");
  
  auto posi=ar.position;
  printf("\nname:%s\nposition:(%lf,%lf,%lf)\n",kn.known["test"]->key.c_str(),posi.x,posi.y,posi.z);
  
  printf("\nnear:\n");
  
  kn.getnear(ar.position,[](ksphere::knowledge*p,void*){
    auto posi=p->obj.position;
    printf("position:(%lf,%lf,%lf)\n",posi.x,posi.y,posi.z);
  },1000,NULL);
  
  double bin[16];
  for(int i=0;i<16;i++){
    bin[i]=0;
  }
  kn.toArray(bin,16,kn.known["test2"]);
  //如果你需要使用二进制的数据，调用vec2bin就行了
  printf("bin:");
  for(int i=0;i<16;i++){
    printf("%f,",(bin[i]));
  }
  printf("\n");
  for(auto it : kn.axionlist)
    printf("axion:\n\t%s\n\tposition:(%lf,%lf,%lf)\n",
      it->key.c_str(),
      it->obj.position.x,
      it->obj.position.y,
      it->obj.position.z
    );
  return 0;
}