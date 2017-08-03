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
}