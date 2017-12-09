#ifndef atulocher_tctree
#define atulocher_tctree
#include "mempool.hpp"
#include <set>
namespace atulocher{
  #define LENP1 (1.0/3.0)
  #define LENP2 (2.0/3.0)
  class tctree{
    public:
    int minN;    //最小个数
    double minS; //最小比值
    std::set<double> tuples;
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
      tctree * owner;
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
      void getTuple(bool intup=false){
        if(this->num<owner->minN && (!intup))
          return;
        getLeftTuple(intup);
        getRightTuple(intup);
      }
      #define gettp(LW,RW,pre) \
        if(!LW)return;\
        if(LW->v){\
          if(intup)owner->tuples.insert(LW->v->position);\
          return;\
        }\
        if(LW->num==0)\
          return;\
        else{\
          if(!intup){\
            if(RW && RW->v==NULL){\
              if(RW->num==0)\
                LW->getTuple(true);\
              else\
              if((((double)LW->num)/((double)RW->num))>owner->minS)\
                LW->getTuple(true);\
              else\
                LW->getTuple(false);\
            }else{\
              LW->getTuple(intup);\
            }\
          }else{\
            if(LW){\
              if(RW){\
                if(RW->num==LW->num){\
                  if(pre)\
                    LW->getTuple(true);\
                }else\
                if(RW->num>LW->num)\
                  RW->getTuple(true);\
                else\
                  LW->getTuple(true);\
              }else\
                LW->getTuple(true);\
            }\
          }\
        }
      void getLeftTuple(bool intup){
        gettp(left,right,true);
      }
      void getRightTuple(bool intup){
        gettp(right,left,false);
      }
      #undef gettp
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
    tctree(double from,double l){
      root=getn();
      root->begin=from;
      root->len=l;
      minN=5;
      minS=0.5;
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