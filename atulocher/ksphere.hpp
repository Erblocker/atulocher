#ifndef atulocher_ksphere
#define atulocher_ksphere
#include "octree.hpp"
#include "rand.hpp"
#include <string>
#include <string.h>
#include <map>
#include <vector>
#include <set>
#include <math.h>
#include <sstream>
#include <iostream>
#include <ctime>
#include <stdio.h>
#include <unistd.h>
#include <sys/file.h>
#include <ctype.h>
/*
* ksphere：
** 一种用向量来学习的方法
** 因其空间布局为球形而得名
*/
namespace atulocher{
  class ksphere{
    typedef octree::vec vec;
    RWMutex locker;
    public:
    octree::octree oct;
    struct knowledge{
      octree::object obj;         //在八叉树中的节点
      bool           isTrue;      //正确性
      bool           isaxion;
      unsigned int   id;
      std::string    key;
      std::string    description; //值
      struct depend{
        knowledge * ptr;
        double w;
        depend()=default;
        depend(const depend&)=default;
        depend(knowledge * p,double iw){
          ptr=p;
          w=iw;
        }
        bool operator>(const depend & d)const{
          return ptr>d.ptr;
        }
        bool operator<(const depend & d)const{
          return ptr<d.ptr;
        }
        bool operator==(const depend & d)const{
          return ptr==d.ptr;
        }
      };
      typedef std::set<depend> dependset;
      dependset dep;
      knowledge(const std::string & k){
        key=k;
        isTrue=true;
        isaxion=false;
        obj.value=this;
        id=0;
        obj.onfree=[](octree::object * self){
          delete (knowledge*)(self->value);
        };
      }
    };
    public:
    inline knowledge * getAxiByKey(const std::string & kk){
      locker.Rlock();
      knowledge * res;
      auto resit=axion.find(kk);
      if(resit==axion.end())
        res=NULL;
      else
        res=resit->second;
      locker.unlock();
      return res;
    }
    inline knowledge * getByKey(const std::string & kk){
      locker.Rlock();
      knowledge * res;
      auto resit=known.find(kk);
      if(resit==known.end())
        res=NULL;
      else
        res=resit->second;
      locker.unlock();
      return res;
    }
    inline void setDepend(knowledge * ob,const std::string & dk,double w){
      if(ob==NULL)return;
      auto r=getAxiByKey(dk);
      if(r==NULL)return;
      knowledge::depend td(r,w);
      ob->dep.insert(td);
    }
    inline void setDepend(const std::string & k1,const std::string & k2,double w){
      if(auto r1=getByKey(k1))
        setDepend(r1,k2,w);
    }
    public:
    std::map<std::string,knowledge*> axion;
    std::map<std::string,knowledge*> known;
    std::vector<knowledge*>          axionlist;
    private:
    int fd;
    bool readconfigline(const char * str){
      octree::vec posi;
      bool t;
      int tb;
      std::string k,k1,k2;
      double w;
      std::string v;
      if(strlen(str)<1)return false;
      std::istringstream iss(str+1);
      if(str[0]=='a'){
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
          axion[k]=kn;
          axionlist.push_back(kn);
          kn->id=axionlist.size()-1;
          kn->isaxion=true;
        }else{
          delete kn;
        }
        return true;
      }else
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
          return false;
        }
        it->second->isTrue=!(it->second->isTrue);
        return true;
      }else
      if(str[0]=='l'){
        iss>>k1;
        iss>>k2;
        iss>>w;
        setDepend(k1,k2,w);
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
      write(fd,buf,strlen(buf));
    }
    void writeconfaxi(
      const std::string & k,
      const std::string & v,
      const octree::vec & p,bool t
    ){
      char buf[4096];
      int tb;
      t==1 ? tb=1 : tb=0;
      snprintf(buf,4096,"a%s %s %f %f %f %d #time:%d\n",
        k.c_str(),
        v.c_str(),
        p.x,
        p.y,
        p.z,
        tb,time(0)
      );
      write(fd,buf,strlen(buf));
    }
    void writeconf(const std::string & k){
      char buf[4096];
      snprintf(buf,4096,"-%s #time:%d\n",k.c_str(),time(0));
      write(fd,buf,strlen(buf));
    }
    void writedep(const std::string & k1,const std::string & k2,double w){
      char buf[4096];
      snprintf(buf,4096,"l%s %s %f #\n",
        k1.c_str(),
        k2.c_str(),
        w
      );
      write(fd,buf,strlen(buf));
    }
    void readconfig(const char * path){
      FILE * fp=NULL;
      fp=fopen(path,"r");
      if(fp==NULL)return;
      char buf[4096];
      while(!feof(fp)){
        bzero(buf,4096);
        fgets(buf,4096,fp);
        confrep(buf);
        if(strlen(buf)<1)continue;
        readconfigline(buf);
      }
      fclose(fp);
    }
    public:
    static void confrep(char * path){
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
      fd=open(path,O_WRONLY|O_CREAT|O_APPEND,0644);
    }
    ~ksphere(){
      if(fd!=-1)close(fd);
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
      std::map<knowledge*,double> dep;
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
      bool mean(const std::string & m,double w,bool limited=true){//w取负数时表示否定
        //总权重必须为1,否则点可能会在球外，引起死循环
        if(w==0)return true;
        knowledge * kkn;
        ks->locker.Rlock();
        auto it=ks->known.find(m);
        octree::vec * pt;
        if(it==ks->known.end()){
          if(readonly){//只读
            ks->locker.unlock();
            return false;
          }else
          if(limited){
            ks->locker.unlock();
            return false;
          }else{
            
            //如果没有
            //记住，没有这个
            octree::vec posi;
          
            ks->locker.unlock();//不然就是die
            ks->addaxion(m,"unknow;",&posi,&kkn);
            ks->locker.Rlock();
          
            pt=&posi;
            dep[kkn]+=w;
          }
        }else{
          pt=&it->second->obj.position;
          kkn=it->second;
          if(kkn->isaxion)
            dep[kkn]+=w;
          else{
            for(auto ita:kkn->dep)
              dep[ita.ptr]+=ita.w*w;
          }
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
        for(auto itd:dep)
          kn->dep.insert(knowledge::depend(itd.first,itd.second));
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
        for(auto itk:dep){
          ks->writedep(key,itk.first->key,itk.second);
        }
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
      return res;
    }
    void getnear(const vec & p,void(*callback)(knowledge*,void*),double range,void *arg,int num=-1){
      vec beg=p-vec(range,range,range);
      vec end=p+vec(range,range,range);
      
      if(!callback)return;
      struct self_o{
        void(*callback)(knowledge*,void*);
        void * arg;
      }self;
      self.arg=arg;
      self.callback=callback;
      
      oct.find([](octree::octreeNode::octval * node,void * s){
        auto self=(self_o*)s;
        self->callback((knowledge*)(node->value),self->arg);
      },beg,end,&self,true,num);
    }
    bool addaxion(const std::string & key,const std::string & value,octree::vec * posi=NULL,knowledge ** okn=NULL){
      locker.Wlock();
      //添加一个基本命题（好像叫做公理）
      if(known[key]!=NULL){
        locker.unlock();
        return false;//已经存在
      }
      //创建节点，不解释
      auto kn=new knowledge(key);
      kn->isaxion=true;
      if(okn)*okn=kn;
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
          octree::vec(100 ,100 , 100)+p,
          &ot,true,1
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
      axion[key]=kn;
      axionlist.push_back(kn);
      kn->id=axionlist.size()-1;
      oct.insert(&(kn->obj));
      writeconfaxi(key,value,p,true);
      locker.unlock();
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
    inline void find(void(*callback)(knowledge*,void*),octree::vec & p,double len,void * arg){
      this->getnear(p,callback,len,arg);
    }
    static void vec2bin(const octree::vec & v,double * d,int l){
      v.GeoHashBin(10000000,d,l);
    }
    inline void toArray(double * arr,int len,const std::string & kn,double imax=1,bool isc=false){
      toArray(arr,len,getByKey(kn),imax,isc);
    }
    template<class T=double>
    void toArray(T * arr,int len,knowledge * kn,double imax=1,bool isc=false){
      if(!kn)return;
      if(isc)
        for(register int i=0;i<len;i++)
          arr[i]=0;
      locker.Rlock();
      for(auto it:kn->dep){
        int id=it.ptr->id;
        if(id>=len)continue;
        arr[id]+=it.w*imax;
      }
      locker.unlock();
    }
    template<class T=double>
    vec loadArray(const T * arr,int len,double imin=0.5){
      vec tmp(0,0,0);
      double s=0;
      locker.Rlock();
      for(register int i=0;i<len;i++){
        
        if(i>=axionlist.size())break;
        if(arr[i]<imin)continue;
        
        s+=arr[i];
        tmp+=axionlist[i]->obj.position*arr[i];
      }
      locker.unlock();
      if(s==0)
        return vec(0,0,0);
      else
        return tmp/s;
    }
  };
}
#endif