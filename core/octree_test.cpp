#include "octree.hpp"
using namespace atulocher;
int main(){
  printf("octree init\n");
  octree::octree oct(
    octree::vec(-300,-300,-300),600.0d
  );
  
  printf("insert\n");
  
  auto o=new octree::object;
  o->onfree=[](octree::object * self){
    printf("free object1\n");
    delete self;
  };
  o->position=octree::vec(15,30,60);
  char cont[]="hello world";
  o->value=cont;
  if(oct.tree->insert(o))printf("insert object1:ok\n");
  
  o=new octree::object;
  o->onfree=[](octree::object * self){
    printf("free object2\n");
    delete self;
  };
  o->position=octree::vec(15,30,61);
  o->value=cont;
  if(oct.tree->insert(o))printf("insert object2:ok\n");
  
  printf("find\n");
  
  oct.tree->find([](octree::object * o,void*){
      printf("(%f,%f,%f):%s\n",
        o->position.x,
        o->position.y,
        o->position.z,
        (char*)o->value
      );
    },
    octree::vec(0,0,0),
    octree::vec(300,300,300),NULL
  );
  oct.tree->find([](octree::object * o,void*){
      printf("(%f,%f,%f):%s\n",
        o->position.x,
        o->position.y,
        o->position.z,
        (char*)o->value
      );
    },
    octree::vec(0,0,0),
    octree::vec(15,30,60.5d),NULL
  );
  return 0;
}