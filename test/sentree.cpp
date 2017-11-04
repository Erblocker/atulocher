#include <iostream>
#include <atulocher/sentree.hpp>
using namespace std;
int main(){
  printf("p1\n");
  atulocher::sentree st;
  printf("p2\n");
  st.add("a",1,0);
  printf("p3\n");
  st.add("b",1,0);
  printf("p4\n");
  st.add("c",1,0);
  printf("p5\n");
  st.paser();
  printf("p6\n");
  st.foreach([](const atulocher::sentree::node * n,void * arg){
    cout<<n->value<<"\t"<<n->index;
    cout<<"\t"<<n->getLink();
    cout<<"\t"<<n->getOffset()<<endl;
    return;
  },NULL);
}