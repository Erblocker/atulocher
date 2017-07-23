#ifndef atulocher_ksphere
#define atulocher_ksphere
#include "octree.hpp"
#include "rand.hpp"
#include <string>
#include <map>
#include <math.h>
/*
* 知识球：
** 一种用向量来学习并进行逻辑思维的方法
** 因其空间布局为球形而得名
*/
namespace atulocher{
  class ksphere{
    octree::octree oct;
    public:
    struct knowledge{
      octree::object obj;         //在八叉树中的节点
      bool           isTrue;      //正确性
      std::string    description; //值
      knowledge(){
        isTrue=true;
      }
    };
    std::map<std::string,knowledge*> known;
    ksphere():oct(
      octree::vec(-10000111.0d,-10000111.0d,-10000111.0d),
      20000222.0d  //设置足够大的范围，以便于贮存数据
    ){
      
    }
    ~ksphere(){
      for(auto it=known.begin();it!=known.end();it++){
        delete (it->second);
      }
    }
    class adder{//添加器
      ksphere * ks;
      public:
      octree::vec position;
      adder(ksphere * k):position(0,0,0){
        ks=k;
      }
      ~adder(){}
      void mean(const std::string & m,double w){//w取负数时表示否定
        auto it=ks->known.find(m);
        if(it==ks->known.end())return;
        position+=it->second->obj.position*w;//坐标乘以权重
      }
      static double m(double a,int b){
        int bf=a/b;
        return a-b*bf;
      }
      bool add(const std::string & key,const std::string & val){
        if(ks->known.find(key)!=ks->known.end())return false;//已经存在
        if(position==octree::vec(0,0,0))return false;
        auto kn=new knowledge();
        kn->description=val;
        ins:
          auto p=position;
          p.x+=m(rand.flo()/100000.0d,30);
          p.y+=m(rand.flo()/100000.0d,30);
          p.z+=m(rand.flo()/100000.0d,30);
          kn->obj.position=p;
        if(!(ks->oct.insert(&(kn->obj))))goto ins;
        ks->known[key]=kn;
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
      //添加一个基本命题（好像叫做公理）
      if(known.find(key)!=known.end())return false;//已经存在
      //创建节点，不解释
      auto kn=new knowledge();
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
    }
    bool negate(const std::string & key){
      auto it=known.find(key);
      if(it==known.end())return false;
      it->second->isTrue=!(it->second->isTrue);
      return true;
    }
  };
}
#endif