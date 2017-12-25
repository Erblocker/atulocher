#ifndef atulocher_triemap
#define atulocher_triemap
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "mempool.hpp"
namespace atulocher{
  class triemap{
    public:
    struct node{
      node * next,
           * parent;
      node * child[256];
      void * data;
      unsigned char position,childnum;
      void construct(){
        parent=NULL;
        for(int i=0;i<256;i++){
          child[i]=NULL;
        }
        data=NULL;
        childnum='\0';
        position='\0';
      }
    };
    mempool<node> npool;
    node * root;
    inline node * getChild(node * n,unsigned char c){
      return n->child[c];
    }
    node * getChild_f(node * n,unsigned char c){
      auto res=n->child[c];
      if(res)
        return res;
      else{
        auto pp=npool.get();
        pp->construct();
        pp->parent=n;
        n->child[c]=pp;
        pp->position=c;
        n->childnum++;
        return pp;
      }
    }
    void autoremove(node * inpn){
      auto n=inpn;
      while(n){
        if(n->data!=NULL)return;
        if(n->childnum!=0)return;
        auto p=n->parent;
        npool.del(n);
        p->childnum--;
        n=p;
      }
    }
    node * find(unsigned char * str){
      auto n=root;
      auto sp=str;
      while(*sp){
        n=getChild(n,*sp);
        
        if(n==NULL)return NULL;
        
        sp++;
      }
      return n;
    }
    node * find_f(unsigned char * str){
      auto n=root;
      auto sp=str;
      while(*sp){
        n=getChild_f(n,*sp);
        sp++;
      }
      return n;
    }
    void erase(unsigned char * str){
      auto p=find(str);
      p->data=NULL;
      autoremove(p);
    }
    void * & operator[](unsigned char * str){
      return find_f(str)->data;
    }
    void destroy(node * n){
      if(n==NULL)return;
      for(int i=0;i<256;i++){
        if(n->child[i])destroy(n->child[i]);
      }
      npool.del(n);
    }
    triemap(){
      root=npool.get();
      root->construct();
    }
    ~triemap(){
      destroy(root);
    }
  };
}
#endif