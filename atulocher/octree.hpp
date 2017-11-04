#ifndef atulocher_octree
#define atulocher_octree
#include "vec3.hpp"
#include "rwmutex.hpp"
#include <stdio.h>
#include <mutex>
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
    void find(void(*callback)(octval*,void*),const vec & beg,const vec & end,void * arg)const{
      vec tbeg=this->orign;
      vec tend;
      tend(
        tbeg.x+this->length,
        tbeg.y+this->length,
        tbeg.z+this->length
      );
      
      if(!AABB(tbeg,tend,beg,end))return;
      for(int i=0;i<8;i++){
        double len=this->length/2.0d;
        
        tbeg=area2vec(i)*len+this->orign;
        tend(
          tbeg.x+len,
          tbeg.y+len,
          tbeg.z+len
        );
        if(AABB(tbeg,tend,beg,end)){
          if(child[i].mode==mode_Node){
            child[i].val.node->find(callback,beg,end,arg);
          }else
          if(child[i].mode==mode_Data){
            auto buf=child[i].val.data;
            if(isinbox(tbeg,buf->position,len)){
              callback(buf,arg);
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
    bool insert(octreeNode::octval * d){
      locker.Wlock();
      //printf("(%f,%f,%f)\n",d->position.x,d->position.y,d->position.z);
      auto r= tree->insert(d);
      locker.unlock();
      return r;
    }
    void find(void(*callback)(octreeNode::octval*,void*),const vec & beg,const vec & end,void * arg){
      locker.Rlock();
      tree->find(callback,beg,end,arg);
      locker.unlock();
    }
  };
  typedef octreeNode::octval object;
}
}
#endif