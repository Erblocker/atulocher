#ifndef atulocher_octree
#define atulocher_octree
#include "vec3.hpp"
#include "rwmutex.hpp"
#include <stdio.h>
#include <mutex>
#include <algorithm>
#define FrontTopLeft      0
#define FrontTopRight     1
#define FrontBottomLeft   2
#define FrontBottomRight  3
#define BackTopLeft       4
#define BackTopRight      5
#define BackBottomLeft    6
#define BackBottomRight   7
#define mode_Empty        0x01
#define mode_Data         0x02
#define mode_Node         0x03
namespace atulocher{
namespace octree{
  typedef vec3<double> vec;
  class octreeNode;
  class Octreepool{
      octreeNode  * freed;
      std::mutex locker;
      unsigned int used,malloced;
      public:
      void del(octreeNode  * f);
      octreeNode  * get();
      Octreepool();
      ~Octreepool();
  };
  class octreeNode{
    friend class Octreepool;
    protected:
    Octreepool * gc;
    octreeNode * gc_next;
    public:
    double length;
    vec    orign;
    int    area;
    unsigned int deep;
    struct octval{
      void(*onfree)(octval*);
      void        * value;
      vec           position;
      octreeNode  * parent;
      int           area;
    };
    struct node_b{
      union{
        octval * data;
        octreeNode  * node;
      }    val;
      char mode;
      node_b(){
        mode=mode_Empty;
        val.data=NULL;
      }
    }child[8];
    octreeNode  * parent;
    octreeNode(Octreepool * g){
      parent=NULL;
      deep=0;
      gc=g;
    }
    static const vec & area2vec(int ar){
      static const vec alist[]={
        vec(0,0,0),
        vec(0,0,1),
        vec(0,1,0),
        vec(0,1,1),
        vec(1,0,0),
        vec(1,0,1),
        vec(1,1,0),
        vec(1,1,1),
      };
      if(ar>=8 || ar<0)
        return alist[0];
      else
        return alist[ar];
    }
    int divSpace(vec & point){
      vec vertex(length,length,length);
      vec p=point-orign;
      int ar=divSpace(vertex,p);
      return ar;
    }
    static int divSpace(vec & vertex,vec & point){
      vec center(
        vertex.x/2.0d,
        vertex.y/2.0d,
        vertex.z/2.0d
      );
      if(center.x>point.x){
        if(center.y>point.y){
          if(center.z>point.z){
            return FrontTopLeft;
          }else{
            return FrontTopRight;
          }
        }else{
          if(center.z>point.z){
            return FrontBottomLeft;
          }else{
            return FrontBottomRight;
          }
        }
      }else{
        if(center.y>point.y){
          if(center.z>point.z){
            return BackTopLeft;
          }else{
            return BackTopRight;
          }
        }else{
          if(center.z>point.z){
            return BackBottomLeft;
          }else{
            return BackBottomRight;
          }
        }
      }
    }
    static bool isinbox(vec & o,vec & p,double length){
      if(p.x<o.x || p.y<o.y || p.z<o.z) return false;
      if(
        p.x>(o.x+length) ||
        p.y>(o.y+length) ||
        p.z>(o.z+length)
      )
        return false;
      return true;
    }
    bool isinbox(vec & p){
      return isinbox(orign,p,length);
    }
    void createNode(int area){
      if(area<0 || area>7)return;
        if(child[area].mode==mode_Empty){
          child[area].mode=mode_Node;
          auto v=gc->get();
          child[area].val.node=v;
          v->length=this->length/2.0d;
          
          vec ori   =(area2vec(area)*(v->length))+this->orign;
          v->orign  =ori;
          v->parent =this;
          v->deep   =this->deep+1;
          v->area   =area;
        }
      
    }
    void remove(int p){
      if(child[p].mode==mode_Empty)return;
      if(child[p].mode==mode_Node){
        child[p].val.node->cleanNode();
        gc->del(child[p].val.node);
        child[p].val.node=NULL;
      }else{
        if(child[p].mode==mode_Data){
          if(child[p].val.data->onfree){
            child[p].val.data->onfree(child[p].val.data);
          }
        }
      }
      child[p].val.data=NULL;
      child[p].mode=mode_Empty;
    }
    void cleanNode(){
      for(int i=0;i<8;i++){
        remove(i);
      }
    }
    bool insert(octval * d,int maxdeep=64){
      if(maxdeep<=0)return false;
      if(d==NULL)return false;
      if(!isinbox(d->position))return false;
      int ar=divSpace(d->position);
      
      if(child[ar].mode==mode_Data){
        if(child[ar].val.data->position==d->position){
          return false;
        }
        auto buf=child[ar].val.data;
        child[ar].mode=mode_Empty;
        
        createNode(ar);
        
        child[ar].val.node->insert(buf,maxdeep-1);
        return child[ar].val.node->insert(d,maxdeep-1);
      
      }else //isnode
      if(child[ar].mode==mode_Node){
        return child[ar].val.node->insert(d);
      
      }else{//Empty
        child[ar].val.data=d;
        d->parent=this;
        d->area=ar;
        child[ar].mode=mode_Data;
        return true;
      
      }
    }
    void autoclean(){
        for(int i=0;i<8;i++)
          if(child[i].mode!=mode_Empty)return;
        
        auto p=parent;
        p->child[area].mode=mode_Empty;
        p->child[area].val.node=NULL;
        gc->del(this);
        p->autoclean();
    }
    static bool AABB(const vec & abeg,const vec & aend,const vec & bbeg,const vec & bend){
      if(aend.x < bbeg.x || abeg.x > bend.x)return false;
      if(aend.y < bbeg.y || abeg.y > bend.y)return false;
      if(aend.z < bbeg.z || abeg.z > bend.z)return false;
      return true;
    }
    void erase(octval * o){
      if(o->parent==NULL)return;
      o->parent->remove(o->area);
    }
    void getchildlist(int * chs,const vec & beg,const vec & end,const vec & nori)const{
      
      vec center=(beg+end)/2.0;
      struct Chl{
        int index;
        double length;
      }chl[8];
      Chl * pl[8];
      
      for(int i=0;i<8;i++){
        chl[i].index=i;
        pl[i]=&chl[i];
        vec p=nori+(area2vec(i)*(this->length/2.0));
        vec b(
          p.x-center.x,
          p.y-center.y,
          p.z-center.z
        );
        chl[i].length=(b.x*b.x)+(b.y*b.y)+(b.z*b.z);
        //平方是单调函数，并且肯定是非负数，所以就不用sqrt了
      }
      
      std::sort(pl,pl+8,[](Chl * p1,Chl * p2){
        return p1->length < p2->length;
      });
      
      for(int i=0;i<8;i++){
        chs[i]=pl[i]->index;
      }
    }
    void find_if(
      void(*callback)(octval*,void*),
      bool(*cond)(const vec & ,void*),
      void * arg
    )const{
      vec tbeg=this->orign;
      vec tcen(
        tbeg.x+this->length/2.0,
        tbeg.y+this->length/2.0,
        tbeg.z+this->length/2.0
      );
      
      //if(!cond(tcen,arg))return;
      
      for(int i=0;i<8;i++){
        double len=this->length/4.0d;
        
        tbeg=area2vec(i)*len+this->orign;
        tcen(
          tbeg.x+len,
          tbeg.y+len,
          tbeg.z+len
        );
        if(cond(tcen,arg)){
          if(child[i].mode==mode_Node){
            child[i].val.node->find_if(callback,cond,arg);
          }else
          if(child[i].mode==mode_Data){
            auto buf=child[i].val.data;
            if(cond(buf->position,arg)){
              callback(buf,arg);
            }
          }else{
          }
        }else{
          continue;
        }
      }
    }
    void find(void(*callback)(octval*,void*),const vec & beg,const vec & end,void * arg,bool issort,int lim)const{
      if(lim==0)return;
      int rl=lim;
      vec tbeg=this->orign;
      vec tend(
        tbeg.x+this->length,
        tbeg.y+this->length,
        tbeg.z+this->length
      );
      
      int chs[8]={0,1,2,3,4,5,6,7};
      if(issort){
        getchildlist(chs,beg,end,((tbeg+tend)/2.0));
      }
      
      //if(!AABB(tbeg,tend,beg,end))return;
      
      for(int j=0;j<8;j++){
        int i=chs[j];
        
        double len=this->length/2.0d;
        
        tbeg=area2vec(i)*len+this->orign;
        tend(
          tbeg.x+len,
          tbeg.y+len,
          tbeg.z+len
        );
        if(AABB(tbeg,tend,beg,end)){
          if(child[i].mode==mode_Node){
            child[i].val.node->find(callback,beg,end,arg,issort,rl);
          }else
          if(child[i].mode==mode_Data){
            auto buf=child[i].val.data;
            if(isinbox(tbeg,buf->position,len)){
              if(rl==0)return;
              callback(buf,arg);
              rl--;
            }
          }else{
            
          }
        }else{
          continue;
        }
      }
    }
  };
  Octreepool::Octreepool(){
        freed=NULL;
        used=0;
        malloced=0;
  }
  Octreepool::~Octreepool(){
        octreeNode  * it1;
        octreeNode  * it=freed;
        while(it){
          it1=it;
          it=it->gc_next;
          delete it1;
        }
  }
  octreeNode  * Octreepool::get(){
        locker.lock();
        used++;
        if(freed){
          octreeNode  * r=freed;
          freed=freed->gc_next;
          locker.unlock();
          r->gc_next=NULL;
          
          (*r)=octreeNode(this);
          return r;
        }else{
          malloced++;
          locker.unlock();
          return new octreeNode(this);
        }
  }
  void Octreepool::del(octreeNode  * f){
        locker.lock();
        //f->destruct();
        f->gc_next=freed;
        freed=f;
        used--;
        locker.unlock();
  }
  class octree{
    Octreepool gc;
    RWMutex locker;
    public:
    octreeNode  * tree;
    octree(vec ori,double len){
      tree=gc.get();
      tree->orign=ori;
      tree->length=len;
    }
    ~octree(){
      tree->cleanNode();
      gc.del(tree);
    }
    inline bool insert(octreeNode::octval * d){
      locker.Wlock();
      auto r= tree->insert(d);
      locker.unlock();
      return r;
    }
    inline void find(void(*callback)(octreeNode::octval*,void*),const vec & beg,const vec & end,void * arg,bool issort=true,int lim=-1){
      locker.Rlock();
      tree->find(callback,beg,end,arg,issort,lim);
      locker.unlock();
    }
    inline void find_if(
      void(*callback)(octreeNode::octval*,void*) ,
      bool(*cond)(const vec & ,void*),
      void * arg
    ){
      locker.Rlock();
      tree->find_if(callback,cond,arg);
      locker.unlock();
    }
    void findInLine(
      void(*callback)(octreeNode::octval*,void*),
      const vec & q,//起点
      const vec & s,//方向
      double range, //射程
      double R,
      void * arg
    ){
      struct sf{
        vec q;
        vec s;
        void * arg;
        double range;
        double R;
        void(*callback)(octreeNode::octval*,void*);
      }self;
      self.callback=callback;
      self.q=q;
      self.s=s;
      self.arg=arg;
      self.R=R;
      self.range=range;
      
      this->find_if(
        [](octreeNode::octval * v,void * a){
          auto self=(sf*)a;
          self->callback(v,self->arg);
        },
        [](const vec & v,void * a)->bool{
          auto self=(sf*)a;
          if(v.length2(self->q)>(self->range*self->range))
            return false;
          
          double a_s=self->s.norm();
          if(a_s==0)return true;
          double a_q=((self->q-v)*self->s).norm();
          auto len=a_q/a_s;
          
          if(len>(self->R))
            return false;
          return true;
        },
        &self
      );
    }
  };
  typedef octreeNode::octval object;
}
}
#endif