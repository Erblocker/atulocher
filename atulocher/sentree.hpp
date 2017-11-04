#ifndef atulocher_sentree
#define atulocher_sentree
#include <stdio.h>
#include <list>
#include <string>
#include <vector>
#include <exception>
#include <sstream>
#include <iostream>
#include <crfpp.h>
#include <memory.h>
#include "mempool.hpp"
namespace atulocher{
  using namespace std;
  class sentree{
    //依存句法树
    public:
    typedef enum{
      LEFT,RIGHT
    }Area;
    
    class ArrayError:public std::exception{};
    class TreeLenError:public std::exception{};
    class MemoryError:public std::exception{};
    
    public:
    class node{
      public:
      typedef mempool_auto<node> GC;
      node *  next,
           *  last,
           *  parent,
           *  gc_next;
      typedef void(*callback)(const node*,void *);
      class List{
        public:
        node *  begin,
             *  end,
             *  parent;
        Area    area;
        void construct(){
          begin=NULL;
          end=NULL;
          parent=NULL;
        }
        int  size()const{
          if(!begin)return 0;
          int s=0;
          auto p=begin;
          while(p){
            s++;
            p=p->next;
          }
          return s;
        }
        void   pushBegin(node * n){
          if(begin){
            n->next=begin;
            begin->last=n;
            n->last=NULL;
            begin=n;
          }else{
            n->last=NULL;
            n->next=NULL;
            begin=n;
            end=n;
          }
          n->parent=this->parent;
          n->area  =this->area;
        }
        void   pushEnd(node * n){
          if(end){
            n->last=end;
            end->next=n;
            n->next=NULL;
            end=n;
          }else{
            n->last=NULL;
            n->next=NULL;
            begin=n;
            end=n;
          }
          n->parent=this->parent;
          n->area  =this->area;
        }
        inline node * popBegin(){
          if(begin){
            return begin->removethis();
          }else
            return NULL;
        }
        inline node * popEnd(){
          if(end){
            return end->removethis();
          }else
            return NULL;
        }
        void foreach(callback cb,void * arg)const{
          if(begin){
            auto p=begin;
            while(p){
              p->foreach(cb,arg);
              p=p->next;
            }
          }
        }
        inline void paserOffsetAll(){
          if(begin){
            auto p=begin;
            while(p){
              p->paserOffset();
              p=p->next;
            }
          }
        }
      }left,right;
      Area    area;
      int     index;
      int     leftoffset,
              rightoffset,
              offset,
              absposi;
      string  value;
      string  senTag;
      string  wordTag;
      string  wordTagFull;
      
      GC * gc;
      node(){
        gc_next=NULL;
      }
      void construct(GC * g){
        gc       =g;
        next     =NULL;
        last     =NULL;
        parent   =NULL;
        index    =0;
        senTag   ="";
        wordTag  ="";
        left.construct();
        right.construct();
        left.parent  =this;
        right.parent =this;
        left.area    =LEFT;
        right.area   =RIGHT;
        offset       =0;
        leftoffset   =0;
        rightoffset  =0;
        absposi      =-1;
      }
      public:
      int getNextLength()const{
        int s=0;
        auto p=next;
        while(p){
          s++;
          p=p->next;
        }
        return s;
      }
      int getLastLength()const{
        int s=0;
        auto p=last;
        while(p){
          s++;
          p=p->last;
        }
        return s;
      }
      int getOffset()const{
        if(parent){
          if(area==LEFT){
            return  getNextLength()+1;
          }else{
            return -getLastLength()-1;
          }
        }else{
          return 0;
        }
      }
      void foreach(callback cb,void * arg)const{
        left.foreach (cb,arg);
        cb(this,arg);
        right.foreach(cb,arg);
      }
      int  getLink()const{
        if(parent)return parent->index;
        return -1;
      }
      char * toString(char * str,int len)const{
        snprintf(str,len,"%d %s %s %s %d_%s",
          this->index,
          this->value.c_str(),
          this->wordTag.c_str(),
          this->wordTagFull.c_str(),
          this->getOffset(),
          this->senTag.c_str()
        );
        return str;
      }
      const char * emptyLine(){
        const static char * el="_ _ _ _ _";
        return el;
      }
      public:
      List * a(Area ar){
        if(ar==LEFT)
          return &left;
        else
          return &right;
      }
      node * removethis(){
        if(last){
          last->next=next;
          next=NULL;
        }else{
          if(parent){
            parent->a(area)->begin=next;
          }
        }
        if(next){
          next->last=last;
          last=NULL;
        }else{
          if(parent){
            parent->a(area)->end=last;
          }
        }
        parent=NULL;
        return this;
      }
      void   pushNext(node * n){
        n->next=next;
        n->last=this;
        if(!next){
          if(parent)
            parent->a(area)->end=n;
        }else
          next->last=n;
        this->next=n;
        n->parent=parent;
      }
      void   pushLast(node * n){
        n->last=last;
        n->next=this;
        if(!last){
          if(parent)
            parent->a(area)->begin=n;
        }else
          last->next=n;
        this->last=n;
        n->parent=parent;
      }
      inline node * popNext(){
        if(next){
          return next->removethis();
        }else
          return NULL;
      }
      inline node * popLast(){
        if(last){
          return last->removethis();
        }else
          return NULL;
      }
      void pull(Area ar,bool pullpar=true){
        if(pullpar){
          if(this->area==RIGHT && ar==LEFT){
            if(last==NULL)return;
          }else
          if(this->area==LEFT  && ar==RIGHT){
            if(next==NULL)return;
          }
          parent->pull(this->area);
        }
        node *p;
        if(ar==LEFT){
          if(last){
            p=this->popLast();
            if(p)
              this->left.pushBegin(p);
            else
              return;
          }else
            return;
        }else{
          if(next){
            p=this->popNext();
            if(p)
              this->right.pushEnd(p);
            else
              return;
          }else
            return;
        }
      }
      inline void folder(int n,Area ar,bool pullpar=true){
        for(int i=0;i<n;i++){
          this->pull(ar,pullpar);
        }
      }
      void paser(){
        paser(&left);
        paser(&right);
      }
      inline void paser(List * s){
        if(s->begin){
          auto p=s->begin;
          while(p){
            if(p->leftoffset>0){
              p->folder(p->leftoffset,LEFT);
              p->leftoffset=0;
              p->paser();
            }
            if(p->rightoffset>0){
              p->folder(p->rightoffset,RIGHT);
              p->rightoffset=0;
              p->paser();
            }
            
            p=p->next;
          }
        }
      }
      node * atOffset(int off){
        if(off==0)return this;
        int ao;
        int i=0;
        if(off>0){
          ao=off;
          auto p=this;
          while(p){
            if(p->next==NULL)
              return p;
            if(i==ao)
              return p;
            p=p->next;
            i++;
          }
        }else{
          ao=-off;
          auto p=this;
          while(p){
            if(p->last==NULL)
              return p;
            if(i==ao)
              return p;
            p=p->last;
            i++;
          }
        }
      }
      void paserOffset(){
        if(offset==0)return;
        int i,ao;
        if(offset>0){
          ao=offset;
          this->offset=0;
          if(next){
            beginr:
            i=1;
            auto p=next;
            while(p){
              if(i==ao || p->next==NULL){
                p->folder(i,LEFT,false);
                return;
              }
              if(p->offset!=0){
                p->paserOffset();
                goto beginr;
              }
              p=p->next;
              i++;
            }
          }else{
            this->offset=0;
            return;
          }
        }else{
          ao=-offset;
          this->offset=0;
          if(last){
            beginl:
            auto p=last;
            i=1;
            while(p){
              if(i==ao || p->last==NULL){
                p->folder(i,RIGHT,false);
                return;
              }
              if(p->offset!=0){
                p->paserOffset();
                goto beginl;
              }
              p=p->last;
              i++;
            }
          }else{
            this->offset=0;
            return;
          }
        }
      }
      static char * cutByFirst_(char * str){
        auto p=str;
        while(*p){
          if(*p=='_'){
            *p='\0';
            return p+1;
          }
          p++;
        }
        return p;
      }
      void loadSenString(const char * str){
        char buf[512];
        snprintf(buf,512,"%s",str);
        char * tag=cutByFirst_(buf);
        this->offset=atoi(buf);
        this->senTag=tag;
      }
      char * getSenString(char * str,int len){
        snprintf(str,len,"%s %s %s",
          value.c_str(),
          wordTag.c_str(),
          wordTagFull.c_str()
        );
        return str;
      }
      void loadTagString(const char * str){
        char buf[512];
        snprintf(buf,512,"%s",str);
        char * full=cutByFirst_(buf);
        this->wordTag=buf;
        this->wordTagFull=full;
      }
      char * getTagString(char * str,int len)const{
        snprintf(str,len,"%s %s_%s",
          value.c_str(),
          wordTag.c_str(),
          wordTagFull.c_str()
        );
        return str;
      }
      void paserOffsetAll(){
        left.paserOffsetAll();
        right.paserOffsetAll();
      }
      private:
      bool paserAbsposiLeft(){
        int i=1;
        node * p=last;
        while(p){
          if(p->index==absposi){
            p->folder(i,RIGHT,false);
            return true;
          }
          p=p->last;
          i++;
        }
        return false;
      }
      bool paserAbsposiRight(){
        int i=1;
        node * p=next;
        while(p){
          if(p->index==absposi){
            p->folder(i,LEFT,false);
            return true;
          }
          p=p->next;
          i++;
        }
        return false;
      }
      public:
      void paserAbsposi(){
        if(absposi<=0)return;
        
        if(absposi>index){
          if(!paserAbsposiRight())paserAbsposiLeft();
        }else{
          if(!paserAbsposiLeft() )paserAbsposiRight();
        }
        absposi=0;
      }
    };
    
    private:
    inline node * newNode(){
      auto r=npool.get();
      r->construct(&npool);
      return r;
    }
    
    private:
    node::GC npool;
    int index;
    
    public:
    inline void bind(mempool<node>* p){
      npool.bind(p);
    }
    
    public:
    node * root;
    list<node*> allnode;
    virtual void init(){
      root=newNode();
      root->value="root";
      index=1;
      //allnode.push_back(root);
    }
    virtual void clear(){
      allnode.clear();
      npool.clear();
      this->init();
    }
    sentree(){
      this->init();
    }
    ~sentree(){
      
    }
    virtual void add(node * p){
      root->right.pushEnd(p);
      allnode.push_back(p);
    }
    virtual void add(
      string v,
      int leftoffset=0,
      int rightoffset=0,
      int offset=0,
      int absposi=-1,
      string mode="",
      string tag="",
      string tagf=""
    ){
      auto p=newNode();
      p->index =this->index;
      this->index++;
      
      p->leftoffset  =leftoffset;
      p->rightoffset =rightoffset;
      p->offset      =offset;
      p->absposi     =absposi;
      
      p->value  =v;
      p->wordTag=tag;
      p->wordTagFull=tagf;
      p->senTag =mode;
      
      root->right.pushEnd(p);
      allnode.push_back(p);
    }
    virtual void loadConllLine(string line){
      istringstream iss(line);
      int index,absposi;
      string wd1,wd2,wt1,wt2,st,m;
      iss>>index;
      iss>>wd1;
      iss>>wd2;
      iss>>wt1;
      iss>>wt2;
      iss>>m;
      iss>>absposi;
      iss>>st;
      this->add(
        wd1,0,0,0,absposi,st,wt1,wt2
      );
    }
    virtual void paser(){
      root->paser();
    }
    virtual void foreach(node::callback cb,void * arg,bool usel=true)const{
      if(usel)
        for(auto p:allnode)
          cb(p,arg);
      else
      root->foreach(cb,arg);
    }
    virtual void paserOffset(){
      root->paserOffsetAll();
    }
    virtual void paserAbsposi(){
      for(auto p:allnode){
        p->paserAbsposi();
      }
    }
    virtual void buildTag(const list<string> & words,CRFPP::Tagger *tagger){
      tagger->clear();
      for(auto wd:words)tagger->add(wd.c_str());
      if (! tagger->parse()) return;
      for (size_t i = 0; i < tagger->size(); ++i) {
        auto p=this->newNode();
        p->value=tagger->x(i, 0);
        p->loadTagString(tagger->y2(i));
        this->add(p);
      }
    }
    virtual void buildTree(CRFPP::Tagger *tagger){
      tagger->clear();
      char buf[512];
      for(auto p:allnode){
        tagger->add(p->getSenString(buf,512));
      }
      if (! tagger->parse()) return;
      auto it=allnode.begin();
      for (size_t i = 0; i < tagger->size(); ++i) {
        if(it==allnode.end())break;
        (*it)->loadSenString(tagger->y2(i));
        it++;
      }
      this->paserOffset();
    }
    virtual bool convertConllFile(const char * path,
      void(*cb1)(const char*,void*),
      void(*cb2)(const char*,void*),
      void * arg
    ){
      this->clear();
      FILE * fp=fopen(path,"r");
      char buf[512];
      if(!fp)return false;
      struct self_o{
        void(*cb1)(const char*,void*);
        void(*cb2)(const char*,void*);
        void * arg;
      }self;
      self.cb1=cb1;
      self.cb2=cb2;
      self.arg=arg;
      while(!feof(fp)){
        bzero(buf,512);
        fgets(buf,512,fp);
        if(buf[0]==' ')break;
        this->loadConllLine(buf);
        this->foreach([](const node * p,void * s){
          if(p->index==0)return;
          char buf[256];
          auto self=(self_o*)s;
          if(self->cb1)
            self->cb1(p->toString(buf,256),self->arg);
          if(self->cb2)
            self->cb2(p->getTagString(buf,256),self->arg);
        },&self);
        
        if(cb1)cb1("_ _ _ _ _",arg);
        if(cb2)cb2("_ _",arg);
      }
      fclose(fp);
      return true;
    }
  };
}
#endif