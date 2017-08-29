#include <atulocher/NN.hpp>
using namespace atulocher::NN;
int main(){
  printf("load music.ann\n");
  auto ann=LoadLayer("music.ann");
  SaveLayer(ann,"test.ann");
  DestroyLayer(ann);
}
