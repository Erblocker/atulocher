#ifndef atulocher_matht
#define atulocher_matht
#include "mempool.hpp"
#include "utils.hpp"
#include "vec3.hpp"
#include <exception>
#include <math.h>
namespace atulocher{
  class matht{
    //library:math_t
    //数学式类
    public:
    
    //error
    class NodeNoFound  :public std::exception{};
    class NodeExist    :public std::exception{};
    class OwnerNoFound :public std::exception{};
    class UnknowValue  :public std::exception{};
    class UnExcNum     :public std::exception{};
    
    struct simple{
      double up,
             down;
      char unknowValue;
      double val()const{
        if(down==0){
          throw UnExcNum();
          return 0;
        }
        return (up/down);
      }
      bool operator==(const simple & n)const{
        if(down==n.down){
          if(down==0)return true;//无穷大等于无穷大
          if(
            up==n.up &&
            unknowValue==n.unknowValue
          )return true;
        }
        if(unknowValue!=n.unknowValue)
          return false;
        if(val()==n.val())
          return true;
        return false;
      }
      #define checkunknow \
        if(unknowValue){\
          throw UnknowValue();\
          return false;\
        }
      #define checkmax(x,y) \
        if(x->down==0){throw UnknowValue();return false;}\
        if(y->down==0){throw UnknowValue();return false;}\
        auto a=x->up/x->down;\
        auto b=y->up/y->down;\
        return a>b;
      bool operator>(const simple & s)const{
        checkunknow;
        checkmax(this,(&s));
      }
      bool operator<(const simple & s)const{
        checkunknow;
        checkmax((&s),this);
      }
      #undef checkunknow
      #undef checkmax
      
      simple(){
        up=0;
        down=1;
        unknowValue='\0';
      }
      simple(const simple&)=default;
      simple(double n){
        up=n;
        down=1;
      }
      simple & operator=(double n){
        up=n;
        down=1;
        return *this;
      }
    };
    class number:public vec3<simple>{
      
    };
    class element{//式
      friend class matht;
      friend class mempool<element>;
      friend class mempool_block<element>;
      
      protected://mempool depended
      element * next,* gc_next;
      
      public:
      matht * owner;
      element * child_left,
              * child_right,
              * parent;
      number value;//数
      
      typedef enum{
        VALUE,
        CHILDREN
      }Mode;
      Mode mode;
      
      class Config{
        public:
        virtual void compute(element*)=0;//对应的算法，加减乘除……
        virtual void computeAll(element*)=0;//完全展开
        virtual void tostring(element*,std::string&)=0;
      }*config;
      
      void setAs(const element * p){
        config=p->config;
        value=p->value;
        mode=p->mode;
      }
      #define autoget(xl,xr) \
        auto p=this;\
        auto last=this;\
        while(p){\
          if(p->xr && p->xr!=last){\
            p=p->xr;\
            while(p){\
              if(p->mode=VALUE)return p;\
              if(p->xl)\
                p=p->xl;\
              else\
                p=p->xr;\
            }\
          }\
          last=p;\
          p=p->parent;\
        }\
        return NULL;
      
      element * left(){
        autoget(child_left,child_right);
      }
      element * right(){
        autoget(child_right,child_left);
      }
      #undef autoget
      
      void foreach(void(*callback)(element*,void*),void * arg){
        if(mode==VALUE){
          callback(this,arg);
          return;
        }
        if(child_left) this->child_left ->foreach(callback,arg);
        if(child_right)this->child_right->foreach(callback,arg);
      }
      
      #define inschn(x) \
        if(mode!=CHILDREN){throw NodeExist();return;}\
        if(x){throw NodeExist();return;}\
        x=p;\
        p->parent=this;
      void insert_left(element * p){
        inschn(child_left);
      }
      void insert_right(element * p){
        inschn(child_right);
      }
      #undef inschn
      
      protected:
      #define checknode \
        if(parent==NULL){\
          throw NodeNoFound();\
          return;\
        }
      #define autopull \
        if(owner==NULL){\
          throw OwnerNoFound();\
          return;\
        }\
        auto p=owner->get();\
        p->mode=CHILDREN;\
        if(parent->child_left==this){\
          parent->child_left=NULL;\
          parent->insert_left(p);\
        }else{\
          parent->child_right=NULL;\
          parent->insert_right(p);\
        }\
        this->parent=NULL;
      
      void pull_left(){
        checknode;
        autopull;
        p->insert_left(this);
      }
      void pull_right(){
        checknode;
        autopull;
        p->insert_right(this);
      }
      #undef autopull
      
      public:
      void insert_this_left(element * p){
        checknode;
        pull_right();
        parent->insert_left(p);
      }
      void insert_this_right(element * p){
        checknode;
        pull_left();
        parent->insert_right(p);
      }
      #undef checknode
      
      inline void compute(){
        this->config->compute(this);
      }
      inline void computeAll(){
        this->config->computeAll(this);
      }
      inline void tostring(std::string & str){
        this->config->tostring(this,str);
      }
      
      void construct(){
        child_left=NULL;
        child_right=NULL;
        parent=NULL;
        mode=VALUE;
      }
      
      void destruct(){
        #define delchild(x) \
          if(x){ \
            x->destruct(); \
          }
        delchild(child_left);
        delchild(child_right);
        owner->del(this);
        #undef delchild
      }
    }*root;
    private:
    mempool<element> gc;
    
    //memory
    public:
    inline element * get(){
      auto p=gc.get();
      p->construct();
      p->owner=this;
      return p;
    }
    private://请直接用element的destruct方法
    inline void del(element * n){
      return gc.del(n);
    }
    
    //construct && destruct
    matht(){
      root=get();
    }
    ~matht(){
      root->destruct();
    }
    virtual void clear(){
      root->destruct();
      root=get();
    }
    
    //iterator
    class iterator{
      public:
      element * ptr;
      iterator(){
        ptr=NULL;
      }
      iterator(const iterator & it){
        ptr=it.ptr;
      }
      iterator & operator=(const iterator & it){
        ptr=it.ptr;
        return *this;
      }
      element * operator->(){
        return ptr;
      }
      bool operator==(const iterator & it)const{
        return ptr==it.ptr;
      }
      iterator & operator++(){
        if(!ptr)return *this;
        ptr=ptr->right();
        return *this;
      }
      iterator & operator--(){
        if(!ptr)return *this;
        ptr=ptr->left();
        return *this;
      }
      iterator operator++(int){
        if(!ptr)return *this;
        auto tmp=*this;
        ptr=ptr->right();
        return tmp;
      }
      iterator operator--(int){
        if(!ptr)return *this;
        auto tmp=*this;
        ptr=ptr->left();
        return tmp;
      }
    };
    #define getiterator(x,y) \
      iterator tmp;\
      auto p=root;\
      while(p){\
        if(p->mode=element::VALUE){\
          tmp.ptr=p;\
          return tmp;\
        }\
        if(p->x)\
          p=p->x;\
        else\
          p=p->y;\
      }\
      return tmp;
    iterator begin(){
      getiterator(child_left,child_right);
    }
    iterator last(){
      getiterator(child_right,child_left);
    }
    #undef getiterator
    iterator end(){
      iterator tmp;
      return tmp;
    }
    
    //operators
    //由于这个库是二叉树，开销非常大，所以operator别乱用
    #define autoclone(x,y) \
      if(p2->x){\
        auto np=get();\
        p1->y(np);\
        clone_node(np,p2->x);\
      }
    void clone_node(element * p1,const element * p2){
      p1->setAs(p2);
      autoclone(child_left ,insert_left);
      autoclone(child_right,insert_right);
    }
    #undef autoclone
    virtual element * clone(const element * p){
      auto re=get();
      clone_node(re,p);
      return re;
    }
    
    //between two object
    #define resetroot \
      auto pr=this->root;\
      pr->parent=NULL;\
      this->root=get();\
      root->mode=element::CHILDREN;
    virtual void put_front(element * p){
      resetroot;
      root->insert_left(p);
      root->insert_right(pr);
    }
    virtual void put_back(element * p){
      resetroot;
      root->insert_right(p);
      root->insert_left(pr);
    }
    #undef resetroot
    
    //一些封装
    virtual element * clone(const matht * p){
      this->clone(p->root);
    }
    virtual element * clone(const matht & p){
      this->clone(p.root);
    }
    virtual void put_front(const matht & p){
      auto pn=clone(p.root);
      put_front(pn);
    }
    virtual void put_back(const matht & p){
      auto pn=clone(p.root);
      put_back(pn);
    }
    virtual void foreach(void(*callback)(element*,void*),void * arg){
      root->foreach(callback,arg);
    }
  };
}
#endif
