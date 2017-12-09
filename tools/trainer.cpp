#include <stdio.h>
#include <string.h>
#include <memory.h>
#include <atulocher/sentree.hpp>
atulocher::mempool<atulocher::sentree::node> gbpool;
void paser(FILE * fp){
  atulocher::sentree st;
  st.bind(&gbpool);
  while(1){
    char buf[4096];
    bzero(buf,4096);
    fgets(buf,4096,fp);
    if(buf[0]==' ')break;
    //std::cout<<buf<<std::endl;
    st.loadConllLine(buf);
  }
  st.paserAbsposi();
  st.foreach([](const atulocher::sentree::node * p,void *){
    if(p->index==0)return;
    char buf[256];
    printf("%s\n",
      p->toString(buf,256)
    );
  },NULL);
  printf("_\t_\t_\t_\t_\n");
}
int main(){
  FILE * fp=fopen("train.conll","r");
  
  if(!fp)return 1;
  while(!feof(fp)){
    paser(fp);
  }
  fclose(fp);
  return 0;
}