#ifndef atulocher_kmeans
#define atulocher_kmeans
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <list>
#include <math.h>
#include <stdlib.h>
#include "mempool.hpp"
#include "vec3.hpp"
namespace atulocher{
  using namespace std;
  class kmeans{
    public:
    int k;
    typedef vector<double> Tuple;//存储每条数据记录
    list<Tuple> * clusters;      //k个簇
    Tuple * means;               //k个中心点
    int dataNum;                 //数据集中数据记录数目
    int dimNum;                  //每条记录的维数
    list<Tuple>     tuples;
    
    kmeans(int ik){
      k=ik;
      clusters=new list<Tuple>[k];
      means=new Tuple[k];
    }
    
    ~kmeans(){
      delete [] clusters;
      delete [] means;
    }
    
    //计算两个元组间的欧几里距离
    double getDistXY(const Tuple& t1, const Tuple& t2){
      double sum = 0;
      for(int i=1; i<=dimNum; ++i){
        sum += (t1[i]-t2[i]) * (t1[i]-t2[i]);
      }
      return sqrt(sum);
    }
    
    //根据质心，决定当前元组属于哪个簇
    virtual int clusterOfTuple(Tuple means[],const Tuple& tuple){
      double dist=getDistXY(means[0],tuple);
      double tmp;
      int label=0;//标示属于哪一个簇
      for(int i=1;i<k;i++){
        tmp=getDistXY(means[i],tuple);
        if(tmp<dist) {dist=tmp;label=i;}
      }
      return label;
    }
    
    //获得给定簇集的平方误差
    virtual double getVar(list<Tuple> clusters[],Tuple means[]){
      double var = 0;
      for (int i = 0; i < k; i++){
        list<Tuple> & t = clusters[i];
        for (Tuple & tp:t){
          var += getDistXY(tp,means[i]);
        }
      }
      //cout<<"sum:"<<sum<<endl;
      return var;
    }
    
    //获得当前簇的均值（质心）
    virtual Tuple getMeans(const list<Tuple>& cluster){
      int num = cluster.size();
      Tuple t(dimNum+1, 0);
      for (auto cp:cluster){
        for(int j=1; j<=dimNum; ++j){
          t[j] += cp[j];
        }
      }
      for(int j=1; j<=dimNum; ++j)
        t[j] /= num;
      return t;
      //cout<<"sum:"<<sum<<endl;
    }
    
    virtual void init(){
      int i=0;
      int tlen=tuples.size();
      vector<list<Tuple>::iterator> tps(tlen);
      auto it=tuples.begin();
      for(int iv=0;iv<tlen;iv++){
        tps[iv]=it;
        it++;
      }
      //一开始随机选取k条记录的值作为k个簇的质心（均值）
      unsigned int seed=time(NULL);
      for(i=0;i<k;){
        int iToSelect = rand_r(&seed)%tps.size();
        seed+=iToSelect;
        if(means[iToSelect].size() == 0){
          Tuple & pit=*(tps[iToSelect]);
          for(int j=0; j<=dimNum; ++j){
            means[i].push_back(pit[j]);
          }
          ++i;
        }
      }
      int lable=0;
      //根据默认的质心给簇赋值
      for(Tuple & tval:tuples){
        lable=clusterOfTuple(means,tval);
        clusters[lable].push_back(tval);
      }
    }
    
    virtual void KMeans(){
      double oldVar=-1;
      double newVar=getVar(clusters,means);
      int t = 0;
      int i,lable;
      while(fabs(newVar - oldVar) >= 1){
        //cout<<"第 "<<++t<<" 次迭代开始："<<endl;
        for (i = 0; i < k; i++){
          means[i] = getMeans(clusters[i]);
        }
        oldVar = newVar;
        newVar = getVar(clusters,means); //计算新的准则函数值
        for (i = 0; i < k; i++){
          clusters[i].clear();
        }
        //根据新的质心获得新的簇
        for(Tuple & tval:tuples){
          lable=clusterOfTuple(means,tval);
          clusters[lable].push_back(tval);
        }
      }
    }
  };
  class kmeans_vec3{
    public:
    typedef vec3<double> vec;
    class node{
      public:
      vec v;
      node * next;
    };
    private:
    mempool<node> pool;
    public:
    int k;
    node        * clusters;      //k个簇
    vec         * means;         //k个中心
    inline double getDistXY(const vec& t1, const vec& t2){
      return sqrt(
        (t1.x-t2.x)*(t1.x-t2.x)+
        (t1.y-t2.y)*(t1.y-t2.y)+
        (t1.z-t2.z)*(t1.z-t2.z)
      );
    }
    int clusterOfTuple(vec means[],const vec& tuple){
      double dist=getDistXY(means[0],tuple);
      double tmp;
      int label=0;//标示属于哪一个簇
      for(int i=1;i<k;i++){
        tmp=getDistXY(means[i],tuple);
        if(tmp<dist) {dist=tmp;label=i;}
      }
      return label;
    }
    
    //获得给定簇集的平方误差
    virtual double getVar(node clusters[],vec means[]){
      double var = 0;
      for (int i = 0; i < k; i++){
        node *tp= &clusters[i];
        while (tp){
          var += getDistXY(tp->v,means[i]);
          tp=tp->next;
        }
      }
      //cout<<"sum:"<<sum<<endl;
      return var;
    }
  };
}
#endif