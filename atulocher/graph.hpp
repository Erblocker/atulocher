#ifndef atulocher_graph
#define atulocher_graph
#include "mempool.hpp"
#include "utils.hpp"
#include <map>
#include <set>
namespace atulocher{
  template<class T_k,class T_v,class T_l>
  class graph{
    public:
    typedef graph<T_k,T_v,T_l> self;
    class node;
    class line{
      friend class mempool<line>;
      friend class mempool_block<line>;
      friend class graph<T_k,T_v,T_l>;
      friend class node;
      protected:
      line * next,
           * gc_next;
      public:
      T_l    value;
      self * owner;
      node * from,
           * to;
      void construct(){
        from=NULL;
        to=NULL;
      }
      void destruct(){}
    };
    protected:
    mempool<line> pool_line;
    public:
    class node{
      friend class mempool<node>;
      friend class mempool_block<node>;
      friend class graph<T_k,T_v,T_l>;
      protected:
      node * next,
           * gc_next;
      public:
      T_v    value;
      self * owner;
      std::set<line*> link,
                      belink;
      void createlink(node * n,T_l & v){
        if(!n)return;
        auto p=owner->createLine();
        p->from=this;
        p->to  =n;
        p->value=v;
        n->belink.insert(p);
        this->link.insert(p);
      }
      void construct(){}
      void destruct(){
        for(auto it:link){
          it->to->belink.remove(it);
          owner->pool_line.del(it);
        }
        for(auto it:belink){
          it->from->link.remove(it);
          owner->pool_line.del(it);
        }
        link.clear();
        belink.clear();
      }
    };
    protected:
    mempool<node> pool_node;
    public:
    void freeNode(node * n){
      if(!n)return;
      n->destruct();
      pool_node.del(n);
    }
    line * createNode(){
      auto p=pool_node.get();
      p->construct();
      p->owner=this;
      return p;
    }
    void freeLine(line * l){
      if(!l)return;
      if(l->from)l->from->link.remove(l);
      if(l->to)  l->to->belink.remove(l);
      l->destruct();
      pool_line.del(l);
    }
    line * createLine(){
      auto p=pool_line.get();
      p->construct();
      p->owner=this;
      return p;
    }
  };
}
#endif