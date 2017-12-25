#ifndef atulocher_kbtree
#define atulocher_kbtree
#include "mempool.hpp"
#include <set>
#include <vector>
#include <math.h>
#include <list>
namespace atulocher{
  class kbtree{
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
    class node;
    class value{
      public:
      vec position;
      void  * data;
      value * next,
            * gc_next;
      node  * parent;
    };
    class node{
      public:
      void construct(){
        left=NULL;
        right=NULL;
        parent=NULL;
        deep=0;
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
           * next;
      value * v;
      kbtree * owner;
      int    deep,k;
      vec    position,
             len;
      void find(void(*callback)(value*,void*),const vec & beg,const vec & end,void * arg,bool issort=true,int mnum=-1)const{
        int maxnum=mnum;
        this->find(callback,beg,end,arg,issort,&maxnum);
      }
      void find(void(*callback)(value*,void*),const vec & beg,const vec & end,void * arg,bool issort,int * maxnum)const{
        vec pend=position;
        for(int i=0;i<owner->k;i++)
          pend[i]+=len[i];
        if(!AABB(position,pend,beg,end,owner->k))return;
        if((*maxnum)==0)return;
        if(v){
          for(int j=0;j<owner->k;j++){
            if(beg[j]>(v->position[j]))return;
            if(end[j]<(v->position[j]))return;
          }
          (*maxnum)--;
          callback(v,arg);
          return;
        }
        if((*maxnum)==0)return;
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
          c1->find(callback,beg,end,arg,issort);
          c2->find(callback,beg,end,arg,issort);
        }else{
          if(left)left->find(callback,beg,end,arg,issort);
          if(right)right->find(callback,beg,end,arg,issort);
        }
      }
      inline double getDiv(int ik){
        return position[k]+(len[k]/2);
      }
      inline int getNextK(){
        if(k+1==owner->k)
          return 0;
        else
          return k+1;
      }
      inline double getLeftBegin(int ck)const{
        return position[ck];
      }
      inline double getRightBegin(int ck)const{
        return position[ck]+(len[ck]/2);
      }
      inline double getChildLen(int ck)const{
        return len[ck]/2;
      }
      inline void getLeftPosition(vec & pv,vec & pl)const{
        int tk=this->k;
        pl=this->len;
        pl[tk]*=0.5;
        pv=this->position;
      }
      inline void getRightPosition(vec & pv,vec & pl)const{
        int tk=this->k;
        pl=this->len;
        pl[tk]*=0.5;
        pv=this->position;
        pv[tk]+=this->len[tk]/2;
      }
      inline double getDiv()const{
        return position[k]+(len[k]/2);
      }
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
      void autoclean(){
        if(v)      return;
        if(left)   return;
        if(right)  return;
        if(!parent)return;
        node * pp;
        if(parent->left==this){
          
          pp=parent;
          pp->left=NULL;
          owner->deln(this);
          pp->autoclean();//this have been deleted
          
        }else
        if(parent->right==this){
          
          pp=parent;
          pp->right=NULL;
          owner->deln(this);
          pp->autoclean();//this have been deleted
          
        }else
        return;
      }
      void insert(value * pv,int maxdeep){
        if(maxdeep<=0)return;
        if(this->v){
          value * pp=this->v;
          this->v=NULL;
          this->insert(pp,maxdeep);
        }
        double dv=getDiv();
        if((pv->position[k])<dv){//in left
          //printf("left: div:%f,posi:%f\n",dv,pv->position[k]);
          if(left){
            left->insert(pv,maxdeep-1);
          }else{
            createLeft();
            left->v=pv;
            pv->parent=left;
          }
        }else{//in right
          //printf("right:div:%f,posi:%f\n",dv,pv->position[k]);
          if(right){
            right->insert(pv,maxdeep-1);
          }else{
            createRight();
            right->v=pv;
            pv->parent=right;
          }
        }
      }
    };
    private:
    mempool<node>       npool;
    mempool_auto<value> vpool;//自动管理内存
    public:
    kbtree(const vec & from,const vec & l,int k){
      init(from,l,k);
    }
    kbtree(int k){
      vec from(k),l(k);
      for(int i=0;i<k;i++){
        from[i]=0;
        l[i]=1;
      }
      init(from,l,k);
    }
    inline void init(const vec & from,const vec & l,int k){
      this->k=k;
      root=getn();
      root->position=from;
      root->len=l;
    }
    ~kbtree(){
      if(root)cleanNode(root);
      root=NULL;
    }
    public:
    node * root;
    private:
    void cleanNode(node * n){
      if(!n)return;
      if(n->left) cleanNode(n->left);
      if(n->right)cleanNode(n->right);
      deln(n);
    }
    inline void deln(node * n){
      npool.del(n);
    }
    inline node * getn(){
      auto p=npool.get();
      p->owner=this;
      p->construct();
      return p;
    }
    public:
    inline value * getv(){
      auto pp=vpool.get();
      pp->position.resize(k);
      return pp;
    }
    inline void insert(value * pv,int maxdeep=128){
      root->insert(pv,maxdeep);
    }
    inline void erase(value * pv){
      if(!pv)return;
      if(pv->parent==NULL)return;
      pv->parent->v=NULL;
      pv->parent->autoclean();
    }
    inline void find(void(*callback)(value*,void*),const vec & beg,const vec & end,void * arg,bool issort=true,int maxnum=-1)const{
      root->find(callback,beg,end,arg,issort,maxnum);
    }
  };
}
#endif