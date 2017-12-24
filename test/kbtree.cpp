#include <atulocher/kbtree.hpp>
using namespace atulocher;
int main(){
  kbtree::vec from(3),len(3),beg(3),end(3);
  
  from[0]=0;
  from[1]=0;
  from[2]=0;
  
  len[0]=100;
  len[1]=100;
  len[2]=100;
  
  beg[0]=25;
  beg[1]=25;
  beg[2]=25;
  
  end[0]=75;
  end[1]=75;
  end[2]=75;
  
  kbtree kt(from,len,3);
  
  auto p1=kt.getv();
  p1->position[0]=50;
  p1->position[1]=50;
  p1->position[2]=50;
  p1->data=(void*)1;
  kt.insert(p1);
  
  printf("insert1\n");

  auto p2=kt.getv();
  p2->position[0]=55;
  p2->position[1]=55;
  p2->position[2]=55;
  p2->data=(void*)2;
  kt.insert(p2);

  printf("insert2\n");
    
  auto p3=kt.getv();
  p3->position[0]=90;
  p3->position[1]=90;
  p3->position[2]=90;
  p3->data=(void*)3;
  kt.insert(p3);
  
  printf("insert3\n");
  
  
  
  kt.root->find([](kbtree::value * v,void*){
    printf("(%f,%f,%f)%d\n",
      v->position[0],
      v->position[1],
      v->position[2],
      v->data
    );
  },beg,end,NULL);

}
