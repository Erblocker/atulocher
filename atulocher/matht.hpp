#ifndef atulocher_matht
#define atulocher_matht
#include "mempool.hpp"
#include "utils.hpp"
#include "vec3.hpp"
#include <exception>
#include <math.h>
#include <stdio.h>
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
      bool iszero()const{
        return up==0;
      }
      bool error()const{
        return down==0;
      }
      bool isone()const{
        if(down==0){
          throw UnExcNum();
          return false;
        }
        return up==down;
      }
      bool operator==(const simple & n)const{
        if(down==n.down){
          if(down==0){
            throw UnExcNum();
            return true;//无穷大等于无穷大
          }
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
        if(x->down==0){throw UnExcNum();return false;}\
        if(y->down==0){throw UnExcNum();return false;}\
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
      
      void zero(){
        up=0;
        down=1;
        unknowValue='\0';
      }
      simple(){
        this->zero();
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
      simple operator-()const{
        simple tmp=*this;
        tmp.up=-up;
        return tmp;
      }
      void tostring(std::string & str)const{
        char buf[256];
        if(down-1.0==0)
          snprintf(buf,256,"%lf",up);
        else
          snprintf(buf,256,"(%lf/%lf)",up,down);
        str=buf;
        if(unknowValue){
          str+='*';
          str+=unknowValue;
        }
      }
      static inline void tosame(simple & p1,simple & p2){
        if(p1.down==p2.down)
          return;
        double d1=p1.down,
               d2=p2.down;
        p1.up*=d2;
        p1.down*=d2;
        p2.up*=d1;
        p2.down*=d1;
      }
      simple operator+(const simple & p)const{
        simple tmp1=*this,
               tmp2=p;
        tosame(tmp1,tmp2);
        tmp1.up+=tmp2.up;
        return tmp1;
      }
      simple operator-(const simple & p)const{
        simple tmp1=*this,
               tmp2=p;
        tosame(tmp1,tmp2);
        tmp1.up-=tmp2.up;
        return tmp1;
      }
      simple operator*(const simple & p)const{
        simple tmp1=*this,
               tmp2=p;
        if(tmp1.error())throw UnExcNum();
        if(tmp2.error())throw UnExcNum();
        tmp1.up*=tmp2.up;
        tmp1.down*=tmp2.down;
        return tmp1;
      }
      simple operator/(const simple & p)const{
        simple tmp1=*this,
               tmp2=p;
        if(tmp1.error()) throw UnExcNum();
        if(tmp2.error()) throw UnExcNum();
        if(tmp2.iszero())throw UnExcNum();
        tmp1.up*=tmp2.down;
        tmp1.down*=tmp2.up;
        return tmp1;
      }
      inline simple & operator+=(const simple & p){
        *this=*this+p;
        return *this;
      }
      inline simple & operator-=(const simple & p){
        *this=*this-p;
        return *this;
      }
      inline simple & operator*=(const simple & p){
        *this=*this*p;
        return *this;
      }
      inline simple & operator/=(const simple & p){
        *this=*this/p;
        return *this;
      }
    };
    class number:public vec3<simple> {
      public:
      void tostring(std::string & str)const{
        str.clear();
        bool havev=false;
        std::string buf;
        
        if(!x.iszero()){havev=true;x.tostring(buf);str+=buf;}
        if(havev)str+="+";
        if(!y.iszero()){havev=true;y.tostring(buf);str+=buf+"*i";}
        if(havev)str+="+";
        if(!z.iszero()){havev=true;z.tostring(buf);str+=buf+"*j";}
      }
      bool isinR()const{
        if(!y.iszero())return false;
        if(!z.iszero())return false;
        return true;
      }
      bool isinN()const{
        if(!isinR())return false;
        if(x.error())return false;
        auto n=x.up/x.down;
        auto nb=floor(n);
        return n==nb;
      }
      inline number & operator+=(const number & p){
        x+=p.x;
        y+=p.y;
        z+=p.z;
        return *this;
      }
      inline number & operator-=(const number & p){
        x-=p.x;
        y-=p.y;
        z-=p.z;
        return *this;
      }
      inline number & zero(){
        x.zero();
        y.zero();
        z.zero();
        return *this;
      }
      inline number operator+(const number & n)const{
        number tmp=*this;
        tmp.x+=n.x;
        tmp.y+=n.y;
        tmp.z+=n.z;
        return tmp;
      }
      inline number operator-(const number & n)const{
        number tmp=*this;
        tmp.x-=n.x;
        tmp.y-=n.y;
        tmp.z-=n.z;
        return tmp;
      }
      inline number operator-()const{
        number tmp=*this;
        tmp.x=-x;
        tmp.y=-y;
        tmp.z=-z;
        return tmp;
      }
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
      
      //element configs
      class Config{
        public:
        virtual void compute(element*)=0;//对应的算法，加减乘除……
        virtual void computeAll(element*)=0;//完全展开
        virtual void tostring(element*,std::string&)=0;
      }*config;
      class config_operator:public Config{
        public:
        std::string stroper;
        virtual void compute(element*){}
        virtual void computeAll(element*){}
        virtual void tostring(element * el,std::string & str){
          std::string buf,bufc;
          int eln=0;
          if(el->child_left){
            ++eln;
            el->child_left->tostring(bufc);
            buf+=bufc;
            buf+=this->stroper;
          }
          if(el->child_right){
            ++eln;
            el->child_right->tostring(bufc);
            buf+=bufc;
          }
          if(eln==2){
            str="(";
            str+=buf;
            str+=")";
          }else{
            str=buf;
          }
        }
      };
      class config_function:public Config{
        public:
        std::string strfunc;
        int argn;
        config_function(){
          argn=2;
        }
        virtual void compute(element*){}
        virtual void computeAll(element * el){
          el->child_left->computeAll();
          el->child_right->computeAll();
          el->compute();
        }
        virtual void tostring_left(element * el,std::string & buf,int & eln){
          if(eln==argn)return;
          std::string bufc;
          if(el->child_left){
            ++eln;
            el->child_left->tostring(bufc);
            buf+=bufc;
            buf+=",";
          }
        }
        virtual void tostring_right(element * el,std::string & buf,int & eln){
          if(eln==argn)return;
          std::string bufc;
          if(el->child_right){
            ++eln;
            el->child_right->tostring(bufc);
            buf+=bufc;
          }
        }
        virtual void tostring(element * el,std::string & str){
          int eln=0;
          str=this->strfunc+"(";
          tostring_left(el,str,eln);
          tostring_right(el,str,eln);
          str+=")";
        }
      };
      class add:public config_operator{
        public:
        add(){
          stroper="+";
        }
        virtual void compute(element * el){
          if(el->child_right && el->child_right->mode==VALUE &&
               el->child_left && el->child_left->mode==VALUE
          ){
              try{
                el->value=el->child_left->value+el->child_right->value;
              }catch(UnExcNum &){
                el->value.zero();
              }
              el->child_left->destruct();
              el->child_right->destruct();
              el->child_left =NULL;
              el->child_right=NULL;
              el->mode=VALUE;
              el->config=NULL;
          }
        }
      };
      class sub:public config_operator{
        public:
        sub(){
          stroper="-";
        }
        virtual void compute(element * el){
          if(el->child_right && el->child_right->mode==VALUE){
            if(el->child_left && el->child_left->mode==VALUE){
              try{
                el->value=el->child_left->value-el->child_right->value;
              }catch(UnExcNum &){
                el->value.zero();
              }
              el->child_left->destruct();
              el->child_right->destruct();
              el->child_left =NULL;
              el->child_right=NULL;
              el->mode=VALUE;
              el->config=NULL;
            }else{
              el->child_right->value=-el->child_right->value;
            }
          }
        }
      };
      template<typename T>
      inline void setconfig(){
        static T defc;
        this->config=&defc;
      }
      //element configs end
      
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
        if(config)this->config->compute(this);
      }
      inline void computeAll(){
        if(config)this->config->computeAll(this);
      }
      void tostring(std::string & str){
        if(mode==CHILDREN)
          this->config->tostring(this,str);
        else
          value.tostring(str);
      }
      void construct(){
        child_left=NULL;
        child_right=NULL;
        parent=NULL;
        mode=VALUE;
        setconfig<add>();
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
    virtual element * createNode(const number & n){
      auto p=get();
      p->value=n;
      p->config=NULL;
      p->mode=element::VALUE;
      return p;
    }
    virtual element * clone(const matht * p){
      return this->clone(p->root);
    }
    virtual element * clone(const matht & p){
      return this->clone(p.root);
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
    virtual void compute(){
      root->compute();
    }
  };
}
#endif
