#ifndef atulocher_ttree
#define atulocher_ttree
#include "mempool.hpp"
namespace atulocher{
  #define LENP1 (1.0/3.0)
  #define LENP2 (2.0/3.0)
  class ttree{
    private:
    //数据结构：三等分二叉树
    //用途：聚类
    class value{
      public:
      double position;
      void  * data;
      value * next,
            * gc_next;
    };
    class node{
      public:
      void construct(){
        left=NULL;
        right=NULL;
        parent=NULL;
        deep=0;
        num=0;
        begin=0;
        len=0;
        v=NULL;
      }
      node * left,
           * right,
           * parent,
           * next,
           * gc_next;
      value * v;
      ttree * owner;
      int    deep,
             num;
      double begin,
             len;
      inline double getChildLen(){
        return (this->len)*LENP2;
      }
      inline double getLeftBegin(){
        return this->begin;
      }
      inline double getRightBegin(){
        return this->begin+(this->len*LENP1);
      }
      inline bool inLeft(double n){
        auto buf=getLeftBegin();
        return (n>buf && n<buf+getChildLen());
      }
      inline bool inRight(double n){
        auto buf=getRightBegin();
        return (n>buf && n<buf+getChildLen());
      }
    };
    mempool_auto<node>  npool;
    mempool_auto<value> vpool;//自动管理内存
    node * root;
    node * getn(){
      auto p=npool.get();
      p->owner=this;
      p->construct();
      return p;
    }
    public:
    ttree(double from,double l){
      root=getn();
      root->begin=from;
      root->len=l;
    }
    private:
    void create_left(node * n){
      if(n->left)return;
      n->left=getn();
      n->left->parent=n;
      n->left->deep=n->deep+1;
      n->left->begin=n->getLeftBegin();
      n->left->len=n->getChildLen();
    }
    void create_right(node * n){
      if(n->right)return;
      n->right=getn();
      n->right->parent=n;
      n->right->deep=n->deep+1;
      n->right->begin=n->getRightBegin();
      n->right->len=n->getChildLen();
    }
    void insert_node(value * v,node * n,bool iadd=true){
      if(iadd)n->num++;
      if(n->v){
        auto buf=n->v;
        n->v=NULL;
        insert_node(buf,n,false);
      }
      if(n->inLeft(v->position)){
        if(n->left)
          insert_node(v,n->left,iadd);
        else{
          create_left(n);
          n->left->v=v;
        }
      }
      if(n->inRight(v->position)){
        if(n->right)
          insert_node(v,n->right,iadd);
        else{
          create_right(n);
          n->right->v=v;
        }
      }
    }
    public:
    void insert(double p,void * d){
      auto pt=vpool.get();
      pt->position=p;
      pt->data=d;
      insert_node(pt,root);
    }
  };
  #undef LENP1
  #undef LENP2
}
#endif