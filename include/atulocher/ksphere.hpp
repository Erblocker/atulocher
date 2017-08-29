#ifndef atulocher_ksphere
#define atulocher_ksphere
#include "octree.hpp"
#include "rand.hpp"
#include <string>
#include <string.h>
#include <map>
#include <math.h>
#include <sstream>
#include <iostream>
#include <ctime>
#include <stdio.h>
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
        obj.onfree=[](octree::object * self){
          delete (knowledge*)(self->value);
        };
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
      //printf("p1");
      if(strlen(str)<1)return false;
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
        //printf("p2\n");
        if(oct.insert(&(kn->obj))){
          //printf("ins\n");
          known[k]=kn;
        }else{
          delete kn;
        }
        return true;
      }else
      if(str[0]=='-'){
        iss>>k;
        //printf("p3");
        auto it=known.find(k);
        //printf("p4");
        if(it==known.end()){
          return false;
        }
        it->second->isTrue=!(it->second->isTrue);
        //printf("p5");
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
      snprintf(buf,4096,"+%s %s %f %f %f %d #time:%d\n",
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
      snprintf(buf,4096,"-%s #time:%d\n",k.c_str(),time(0));
      fwrite(buf,strlen(buf),1,fd);
    }
    void readconfig(const char * path){
      FILE * fp=NULL;
      fp=fopen(path,"r");
      if(fp==NULL)return;
      char buf[4096];
      //printf("rr\n");
      while(!feof(fp)){
        //printf("re\n");
        bzero(buf,4096);
        fgets(buf,4096,fp);
        //printf("rl\n");
        confrep(buf);
        if(strlen(buf)<1)continue;
        //printf("rp\n");
        readconfigline(buf);
      }
      //printf("end\n");
      fclose(fp);
    }
    public:
    static void confrep(char * path){
      //printf("cc\n");
      char * p=path;
      while(*p){
        if(*p=='#' || *p=='\n'){
          *p='\0';
          return;
        }
        p++;
      }
    }
    ksphere()=delete;
    void operator=(ksphere&)=delete;
    ksphere(const char * path):oct(
      octree::vec(-10000111.0d,-10000111.0d,-10000111.0d),
      20000222.0d  //设置足够大的范围，以便于贮存数据
    ){
      srand(time(0));
      readconfig(path);
      //printf("load fd\n");
      fd=fopen(path,"a");
      //printf("lded\n");
    }
    ~ksphere(){
      if(fd)fclose(fd);
    }
    static double randn(){
      if(rand()>(RAND_MAX/2)){
        return  1.0d;
      }else{
        return -1.0d;
      }
    }
    class adder{//添加器
      ksphere * ks;
      int pn;
      public:
      octree::vec position;
      bool readonly;
      adder()=delete;
      adder(ksphere * k):position(0,0,0){
        ks=k;
        pn=0;
        readonly=false;
      }
      ~adder(){}
      void mean(const octree::vec & p,double w){
        position+=p*w;//坐标乘以权重
      }
      bool mean(const std::string & m,double w){//w取负数时表示否定
        //总权重必须为1,否则点可能会在球外，引起死循环
        ks->locker.Rlock();
        auto it=ks->known.find(m);
        octree::vec * pt;
        if(it==ks->known.end()){
          if(readonly){//只读
            ks->locker.unlock();
            return false;
          }
          //如果没有
          //记住，没有这个
          octree::vec posi;
          ks->addaxion(m,"unknow;",&posi);
          pt=&posi;
        }else{
          pt=&it->second->obj.position;
        }
        mean(*pt,w);
        pn++;
        ks->locker.unlock();
        return true;
      }
      static double m(double a,int b){
        int bf=a/b;
        return a-b*bf;
      }
      bool add(const std::string & key,const std::string & val){
        if(readonly)return false;
        if(pn==0)
          return ks->addaxion(key,val);
          //没有任何依据的命题，当然就是公理啦
        ks->locker.Wlock();
        if(ks->known[key]!=NULL){
          ks->locker.unlock();
          return false;//已经存在
        }
        //注：这一部分已经被取消
        //原因是：根本没啥用
        //不合情理的命题还是存在
        //不过基本上都分布于球心附近
        //反正记住，越靠近（0,0,0）的命题越不靠谱
        //搜索时避开就行了
        //////////////////////////////////
        //这里也可以给“理解”这个词语下一个定义了：
        //  能够将彼此之间无关的数据转化为彼此之间能够比较远近距离的数据
        //////////////////////////////////
        //if(position==octree::vec(0,0,0)){
          //ks->locker.unlock();
          //return false;
          ////是空集。
          ////你输入了什么？
          ////又大又小？
          ////又高又矮？
          ////还是啥不符合情理的东西？
          //注：对立命题连线中点当然在球心上
        //}
        auto kn=new knowledge(key);
        kn->description=val;
        ins:
          auto p=position;
          p.x+=rand()/(RAND_MAX/30.0d)*randn();
          p.y+=rand()/(RAND_MAX/30.0d)*randn();
          p.z+=rand()/(RAND_MAX/30.0d)*randn();
          kn->obj.position=p;
        if(!(ks->oct.insert(&(kn->obj))))goto ins;
        ks->known[key]=kn;
        ks->writeconf(key,val,p,true);
        ks->locker.unlock();
        return true;
      }
    };
    octree::vec randposi(){//随机生成一个球面上的点
      beg:
      octree::vec p(
        rand()*randn(),
        rand()*randn(),
        rand()*randn()
      );
      auto norm=p.norm();
      if(norm==0.0d)goto beg;
      auto res=(p/norm)*10000000.0d;
      //printf("%f,%f,%f norm:%f\n",res.x,res.y,res.z,norm);
      return res;
    }
    bool addaxion(const std::string & key,const std::string & value,octree::vec * posi=NULL){
      //printf("p1\n");
      locker.Wlock();
      //printf("r1\n");
      //添加一个基本命题（好像叫做公理）
      if(known[key]!=NULL){
        //printf("have key\n");
        locker.unlock();
        return false;//已经存在
      }
      //printf("hh\n");
      //创建节点，不解释
      auto kn=new knowledge(key);
      kn->description=value;
      struct of_t{
        int num;
      }ot;
      //printf("p2\n");
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
        //printf("p3\n");
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
      //printf("p4\n");
      writeconf(key,value,p,true);
      locker.unlock();
      //printf("p5\n");
      if(posi){
        *posi=p;
      }
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
    static void vec2bin(const octree::vec & v,double * d,int l){
      v.GeoHashBin(10000000,d,l);
    }
  };
}
#endif