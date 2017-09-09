#include <iostream>
#include <atulocher/NN.hpp>
using namespace atulocher::NN;
int main(){
  printf("load music.ann\n");
  auto ann=LoadLayer("music.ann");
  std::string str;
  ann->toString(str);
  std::cout<<str<<std::endl;
  SaveLayer(ann,"test.ann");
  DestroyLayer(ann);
}
