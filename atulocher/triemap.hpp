#ifndef atulocher_triemap
#define atulocher_triemap
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <map>
#include "mempool.hpp"
namespace atulocher{
  template<typename T>
  class triemap{
    public:
    struct node{
      node               * next,
                         * parent;
      std::map<char,node*> child;
      T   *                data;
      char buffer[sizeof(T)];
      unsigned char position;
      void construct(){
        parent=NULL;
        child.clear();
        data=NULL;
        //data=(T*)buffer;
        //*data=T();
        position='\0';
      }
      void destruct(){
        if(data){
          data->~T();
          data=NULL;
        }
        child.clear();
      }
      void operator=(const T & d){
        if(data)data->~T();
        data=(T*)buffer;
        data->T(d);
      }
    };
    mempool<node> npool;
    node * root;
    inline node * getChild(node * n,unsigned char c){
      auto it=n->child.find(c);
      if(it==n->child.end())
        return NULL;
      else
        return it->second;
      //return n->child[c];
    }
    node * getChild_f(node * n,unsigned char c){
      node *& res=n->child[c];
      if(res)
        return res;
      else{
        auto pp=npool.get();
        pp->construct();
        pp->parent=n;
        res=pp;
        pp->position=c;
        return pp;
      }
    }
    void autoremove(node * inpn){
      auto n=inpn;
      while(n){
        if(n->data!=NULL)return;
        if(n->child.size()!=0)return;
        auto p=n->parent;
        n->destruct();
        npool.del(n);
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
      p->data->~T();
      p->data=NULL;
      autoremove(p);
    }
    T & operator[](unsigned char * str){
      auto it=find_f(str);
      if(it->data==NULL)
        it->data=(T*)it->buffer;
      *(it->data)=T();
      return it->data;
    }
    void destroy(node * n){
      if(n==NULL)return;
      if(n->data){
        n->data->~T();
        n->data=NULL;
      }
      for(auto it:n->child){
        if(it.second==NULL)continue;
        destroy(it.second);
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
  triemap<int> test;
}
#endif
