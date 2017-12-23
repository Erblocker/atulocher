#ifndef atulocher_ktctree
#define atulocher_ktctree
#include "mempool.hpp"
#include <set>
#include <vector>
#include <math.h>
#include <list>
#define LENP1 (1.0/3.0)
#define LENP2 (2.0/3.0)
namespace atulocher{
//note:
//三等分树（三等分二叉树）是一种特殊的二叉树
//它将一个区域三等分，两边部分分别属于左右节点，中间部分同时属于两个节点
//=============================================================
  class ktctree{//k维三等分树
    public:
    typedef std::vector<double> vec;
    static double getDistXY(const vec& t1, const vec& t2,int k){//求距离
      double sum = 0;
      for(int i=1; i<=k; ++i){
        sum += (t1[i]-t2[i]) * (t1[i]-t2[i]);
      }
      return sqrt(sum);
    }
    static bool AABB(const vec & abeg,const vec & aend,const vec & bbeg,const vec & bend,int k){
      for(int i=0;i<k;i++)
        if(
          aend[i] < bbeg[i] ||
          abeg[i] > bend[i]
        )
          return false;
      return true;
    }
    int k;
    class value{
      public:
      vec position;
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
        v=NULL;
        if(position.size()!=owner->k)position.resize(owner->k);
        if(len.size()!=owner->k)len.resize(owner->k);
        for(int i=0;i<owner->k;i++){
          position[i]=0;
          len[i]=0;
        }
      }
      node * left,
           * right,
           * parent,
           * next,
           * gc_next;
      value * v;
      ktctree * owner;
      int    deep,
             num,k;
      vec    position,
             len;
      void find(void(*callback)(value*,void*),const vec & beg,const vec & end,void * arg,bool issort=true)const{
        std::set<value*> ud;
        find(callback,beg,end,arg,issort,&ud);
      }
      void find(void(*callback)(value*,void*),const vec & beg,const vec & end,void * arg,bool issort,std::set<value*> * ud)const{
        vec pend=position;
        for(int i=0;i<owner->k;i++)
          pend[i]+=len[i];
        if(!AABB(position,pend,beg,end,owner->k))return;
        if(v){
          if(ud->find(v)==ud->end()){
            ud->insert(v);
            callback(v,arg);
          }
          //callback(v,arg);
          return;
        }
        if(left!=NULL && right!=NULL){
          auto c1=left;
          auto c2=right;
          if(issort){
            vec center(owner->k);
            vec center1(owner->k);
            vec center2(owner->k);
            for(int i=0;i<owner->k;i++){
              center[i] =(beg[i]+end[i])/2;
              center1[i]=c1->position[i]+(c1->len[i]/2);
              center2[i]=c2->position[i]+(c2->len[i]/2);
            }
            double l1=getDistXY(center,center1,owner->k);
            double l2=getDistXY(center,center2,owner->k);
            if(l1>l2){
              register node * tmp=c1;
              c1=c2;
              c2=tmp;
            }
          }
          c1->find(callback,beg,end,arg,issort,ud);
          c2->find(callback,beg,end,arg,issort,ud);
        }else{
          if(left)left->find(callback,beg,end,arg,issort,ud);
          if(right)right->find(callback,beg,end,arg,issort,ud);
        }
      }
      inline double getDiv(){
        return position[k]+(len[k]/2);
      }
      inline int getNextK(){
        if(k+1==owner->k)
          return 0;
        else
          return k+1;
      }
      inline void getLeftPosition(vec & pv,vec & pl)const{
        //int tk=getNextK();
        int tk=this->k;
        pl=this->len;
        pl[tk]*=LENP2;
        pv=this->position;
      }
      inline void getRightPosition(vec & pv,vec & pl)const{
        //int tk=getNextK();
        int tk=this->k;
        pl=this->len;
        pl[tk]*=LENP2;
        pv=this->position;
        pv[tk]+=this->len[tk]*LENP1;
      }
      private:
      inline node * createNode(){
        auto p=owner->getn();
        p->parent=this;
        return p;
      }
      inline void createLeft(){
        if(v)return;
        if(left)return;
        left=createNode();
        getLeftPosition(left->position,left->len);
        left->deep=deep+1;
        left->k=getNextK();
      }
      inline void createRight(){
        if(v)return;
        if(right)return;
        right=createNode();
        getRightPosition(right->position,right->len);
        right->deep=deep+1;
        right->k=getNextK();
      }
      inline void insert_left(value * pv,bool re){
        if(!pv)return;
        if(!inLeft(pv->position))return;
        if(left){
          left->insert(pv,re);
        }else{
          createLeft();
          left->v=pv;
          ++(left->num);
        }
      }
      inline void insert_right(value * pv,bool re){
        if(!pv)return;
        if(!inRight(pv->position))return;
        if(right){
          right->insert(pv,re);
        }else{
          createRight();
          right->v=pv;
          ++(right->num);
        }
      }
      public:
      void insert(value * pv,bool re=true){
        if(deep>=owner->maxdeep)return;
        if(v!=NULL){
          auto pp=v;
          v=NULL;
          insert_left(pp,false);
          insert_right(pp,false);
          
        }else{
          //if(left==NULL && right==NULL){
          //  this->v=pv;
          //  if(re)++num;
          //  return;
          //}
        }
        if(re)++num;
        insert_left(pv,true);
        insert_right(pv,true);
      }
      inline double getChildLen(int ck)const{
        return len[ck]*LENP2;
      }
      inline double getLeftBegin(int ck)const{
        return position[ck];
      }
      inline double getRightBegin(int ck)const{
        return position[ck]+(len[ck]*LENP1);
      }
      inline bool inLeft(double n,int ck)const{
        double lmin=getLeftBegin(ck);
        double lmax=lmin+getChildLen(ck);
        //printf("left: n:%f,lmin:%f,lmax:%f,k:%d\n",n,lmin,lmax,ck);
        return (n>lmin && n<lmax);
      }
      inline bool inLeft(const vec & pv)const{
        //int ck=getNextK();
        int ck=k;
        return inLeft(pv[ck],ck);
      }
      inline bool inRight(double n,int ck)const{
        double lmin=getRightBegin(ck);
        double lmax=lmin+getChildLen(ck);
        //printf("right: n:%f,lmin:%f,lmax:%f,k:%d\n",n,lmin,lmax,ck);
        return (n>lmin && n<lmax);
      }
      inline bool inRight(const vec & pv)const{
        //int ck=getNextK();
        int ck=k;
        return inRight(pv[ck],ck);
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
          if(intup)owner->tuples.push_back(LW->v->position);\
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
    private:
    mempool_auto<node>  npool;
    mempool_auto<value> vpool;//自动管理内存
    public:
    ktctree(const vec & from,const vec & l,int k){
      this->k=k;
      root=getn();
      root->position=from;
      root->len=l;
      minN=5;
      minS=0.5;
      maxdeep=128;
      nodenum=0;
    }
    node * root;
    private:
    int nodenum;
    node * getn(){
      nodenum++;
      auto p=npool.get();
      p->owner=this;
      p->construct();
      return p;
    }
    public:
    inline int getSize(){
      return nodenum;
    }
    inline value * getv(){
      auto pp=vpool.get();
      pp->position.resize(k);
      return pp;
    }
    inline void insert(value * pv){
      root->insert(pv);
    }
    int maxdeep;
    int minN;    //最小个数
    double minS; //最小比值
    std::list<vec> tuples;
  };
}
#undef LENP1
#undef LENP2
#endif