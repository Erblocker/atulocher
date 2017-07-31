#ifndef atulocher_ksphere
#define atulocher_ksphere
#include "octree.hpp"
#include "rand.hpp"
#include <string>
#include <string.h>
#include <map>
#include <math.h>
#include <stdio.h>
#include <sstream>
/*
* 知识球：
** 一种用向量来学习并进行逻辑思维的方法
** 因其空间布局为球形而得名
*/
namespace atulocher{
  class ksphere{
    octree::octree oct;
    RWMutex locker;
    public:
    struct knowledge{
      octree::object obj;         //在八叉树中的节点
      bool           isTrue;      //正确性
      std::string    key;
      std::string    description; //值
      knowledge(const std::string & k){
        key=k;
        isTrue=true;
        obj.value=this;
      }
    };
    private:
    std::map<std::string,knowledge*> known;
    FILE  * fd;
    bool readconfigline(const char * str){
      octree::vec posi;
      bool t;
      int tb;
      std::string k;
      std::string v;
      std::istringstream iss(str+1);
      if(str[0]=='+'){
        iss>>k;
        iss>>v;
        iss>>posi.x;
        iss>>posi.y;
        iss>>posi.z;
        iss>>tb;
        t=(tb==1);
        auto kn=new knowledge(k);
        kn->obj.position=posi;
        kn->description=v;
        kn->isTrue=t;
        if(oct.insert(&(kn->obj))){
          known[k]=kn;
        }else{
          delete kn;
        }
        return true;
      }else
      if(str[0]=='-'){
        iss>>k;
        auto it=known.find(k);
        if(it==known.end()){
          locker.unlock();
          return false;
        }
        it->second->isTrue=!(it->second->isTrue);
        return true;
      }
      return false;
    }
    void writeconf(
      const std::string & k,
      const std::string & v,
      const octree::vec & p,bool t
    ){
      char buf[4096];
      int tb;
      t==1 ? tb=1 : tb=0;
      sprintf(buf,"+%s %s %f %f %f %d #time:%d\n",
        k.c_str(),
        v.c_str(),
        p.x,
        p.y,
        p.z,
        tb,time(0)
      );
      fwrite(buf,strlen(buf),1,fd);
    }
    void writeconf(const std::string & k){
      char buf[4096];
      sprintf(buf,"-%s #time:%d\n",k.c_str(),time(0));
      fwrite(buf,strlen(buf),1,fd);
    }
    static void confrep(char * path){
      auto p=path;
      while(*p){
        if(*p=='#'){
          *p='\0';
          return;
        }
        p++;
      }
    }
    void readconfig(const char * path){
      FILE * fp=NULL;
      fp=fopen(path,"r");
      if(fp==NULL)return;
      char buf[4096];
      while(!feof(fp)){
        fgets(buf,4096,fd);
        confrep(buf);
        readconfigline(buf);
      }
      fclose(fp);
    }
    public:
    ksphere(const char * path):oct(
      octree::vec(-10000111.0d,-10000111.0d,-10000111.0d),
      20000222.0d  //设置足够大的范围，以便于贮存数据
    ){
      readconfig(path);
      fd=fopen(path,"a");
    }
    ~ksphere(){
      fclose(fd);
      for(auto it=known.begin();it!=known.end();it++){
        delete (it->second);
      }
    }
    class adder{//添加器
      ksphere * ks;
      int pn;
      public:
      octree::vec position;
      adder(ksphere * k):position(0,0,0){
        ks=k;
        pn=0;
      }
      ~adder(){}
      void mean(const std::string & m,double w){//w取负数时表示否定
        ks->locker.Rlock();
        auto it=ks->known.find(m);
        if(it==ks->known.end()){
          ks->locker.unlock();
          return;
        }
        position+=it->second->obj.position*w;//坐标乘以权重
        pn++;
        ks->locker.unlock();
      }
      static double m(double a,int b){
        int bf=a/b;
        return a-b*bf;
      }
      bool add(const std::string & key,const std::string & val){
        if(pn==0)
          return ks->addaxion(key,val);
          //没有任何依据的命题，当然就是公理啦
        ks->locker.Wlock();
        if(ks->known.find(key)!=ks->known.end()){
          ks->locker.unlock();
          return false;//已经存在
        }
        //这一部分已经被取消
        //原因是：根本没啥用
        //不合情理的命题还是存在
        //不过基本上都分布于球心附近
        //搜索时避开就行了
        //if(position==octree::vec(0,0,0)){
          //ks->locker.unlock();
          //return false;
          ////是空集。
          ////你输入了什么？
          ////又大又小？
          ////又高又矮？
          ////还是啥不符合情理的东西？
        //}
        auto kn=new knowledge(key);
        kn->description=val;
        ins:
          auto p=position;
          p.x+=m(rand.flo()/100000.0d,30);
          p.y+=m(rand.flo()/100000.0d,30);
          p.z+=m(rand.flo()/100000.0d,30);
          kn->obj.position=p;
        if(!(ks->oct.insert(&(kn->obj))))goto ins;
        ks->known[key]=kn;
        ks->writeconf(key,val,p,true);
        ks->locker.unlock();
        return true;
      }
    };
    octree::vec randposi(){//随机生成一个球面上的点
      octree::vec p(
        (rand.flo()),
        (rand.flo()),
        (rand.flo())
      );
      auto norm=p.invnorm();
      return (p*norm*10000000.0d);
    }
    bool addaxion(const std::string & key,const std::string & value){
      locker.Wlock();
      //添加一个基本命题（好像叫做公理）
      if(known.find(key)!=known.end()){
        locker.unlock();
        return false;//已经存在
      }
      //创建节点，不解释
      auto kn=new knowledge(key);
      kn->description=value;
      struct of_t{
        int num;
      }ot;
      get_position:
        auto p=randposi();//随机生成一个球面坐标
        ot.num=0;
        oct.find(
          [](octree::object * o,void*ot){
            ((of_t*)ot)->num++;
          },
          octree::vec(-100,-100,-100)+p,
          octree::vec(100 ,100 , 100)+p,&ot
        );//搜索附近的点
        if(ot.num>0)goto get_position;//存在，重新产生坐标
        
        ot.num=0;
        oct.find(
          [](octree::object * o,void*ot){
            ((of_t*)ot)->num++;
          },
          octree::vec(-100,-100,-100)-p,
          octree::vec(100 ,100 , 100)-p,&ot
        );
        //搜索对立面的点
        //对立的命题坐标相对于原点对称
        if(ot.num>0)goto get_position;//存在，重新产生坐标
        
      kn->obj.position=p;
      known[key]=kn;
      oct.insert(&(kn->obj));
      writeconf(key,value,p,true);
      locker.unlock();
      return true;
    }
    bool negate(const std::string & key){
      locker.Wlock();
      auto it=known.find(key);
      if(it==known.end()){
        locker.unlock();
        return false;
      }
      it->second->isTrue=!(it->second->isTrue);
      writeconf(key);
      locker.unlock();
      return true;
    }
    knowledge * find(const std::string & k){
      auto it=known.find(k);
      if(it==known.end())
        return NULL;
      else
        return it->second;
    }
    void find(void(*callback)(knowledge*,void*),octree::vec & p,double len,void * arg){
      locker.Rlock();
      struct of_t{
        void(*callback)(knowledge*,void*);
        void * arg;
      }ot;
      ot.callback=callback;
      ot.arg=arg;
      oct.find([](octree::object * obj,void * arg){
          auto self=(of_t*)arg;
          auto kn=(knowledge*)obj->value;
          self->callback(kn,self->arg);
        },
        p-octree::vec(len,len,len),
        p+octree::vec(len,len,len),
        &ot
      );
      locker.unlock();
    }
  };
}
#endif