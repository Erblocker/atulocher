#ifndef atulocher_bayes
#define atulocher_bayes
#include <vector>
#include <list>
#include <map>
#include <string>
#include <sstream>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <exception>
namespace atulocher{
  namespace bayes{
    using namespace std;
    class FloatError:public std::exception{};
    class naive{ //暴力膜蛤不可取！！！
      int s;//样品总数
      struct elm{
          int n;//出现次数
          double p;
          elm(){
            n=0;
            p=1;
          }
          elm(const elm & i){
            n=i.n;
            p=i.p;
          }
          elm & operator=(const elm & i){
            n=i.n;
            p=i.p;
            return *this;
          }
          void update(int s){
            n++;
            p=n/s;
          }
          void init(int bn,int bs){
            n=bn;
            p=n/bs;
          }
      };
      struct cP{
        map<int,elm> l;
        elm p;
      };
      map<int,cP>      c;
      map<int,elm>     sc;
      public:
      void train(const list<int> & arr,int tc){
        
        cP & ptc=c[tc];
        this->s++;
        ptc.p.update(s);
        
        for(auto key:arr){
          
          elm & it=ptc.l[key];
          it.update(s);
          sc[key].update(s);
        }
      }
      int predict(const list<int> & arr,double * getp=NULL)const{
        int out;
        double op=0;
        double bt=1;
        for(auto key:arr){
          auto ite=sc.find(key);
          if(ite==sc.end())continue;
          bt*=ite->second.p;
        }
        if(bt==0){
          throw FloatError();
          return -1;
        }
        for(auto it=c.begin();it!=c.end();it++){
          double bufp=1;
          for(auto key:arr){
            auto ite=it->second.l.find(key);
            if(ite==it->second.l.end())continue;
            const elm & ptc=ite->second;
            bufp*=ptc.p;
          }
          if(bufp>op){
            op=bufp;
            out=it->first;
          }
        }
        if(getp)*getp=op/bt;
        return out;
      }
      bool save(const char * path){
        FILE * fp=fopen(path,"a");
        if(fp==NULL)return false;
        fprintf(fp,"G %d\n",s);
        for(auto it=c.begin();it!=c.end();it++){
          fprintf(fp,"C %d %d\n",
            it->first,
            it->second.p.n
          );
          for(auto lt=it->second.l.begin();lt!=it->second.l.end();lt++){
            fprintf(fp,"P %d %d %d\n",
              it->first,
              lt->first,
              lt->second.n
            );
          }
        }
        for(auto it=sc.begin();it!=sc.end();it++){
          fprintf(fp,"S %d %d\n",
            it->first,
            it->second.n
          );
        }
        fclose(fp);
        return true;
      }
      bool load(const char * path){
        FILE * fp=fopen(path,"r");
        if(fp==NULL)return false;
        char buf[128];
        while(!feof(fp)){
          bzero(buf,128);
          fgets(buf,128,fp);
          istringstream iss(buf);
          string m;
          int k1,k2,bfn;
          iss>>m;
          if(m=="G"){
            iss>>this->s;
          }else
          if(m=="C"){
            iss>>k1;
            iss>>bfn;
            c[k1].p.init(bfn,s);
          }else
          if(m=="P"){
            iss>>k1;
            iss>>k2;
            iss>>bfn;
            c[k1].l[k2].init(bfn,s);
          }else
          if(m=="S"){
            iss>>k1;
            iss>>bfn;
            sc[k1].init(bfn,s);
          }
        }
        fclose(fp);
        return true;
      }
      naive(){
      }
    };
  }
}
#endif
