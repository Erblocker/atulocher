#ifndef atulocher_language
#define atulocher_language
#include <map>
#include <string>
#include <vector>
#include <set>
#include <mutex>
#include <exception>
#include "word2vec.hpp"
#include "NN.hpp"
#include "db.hpp"
namespace atulocher{
  class NNError:public std::exception{
  };
  typedef vec3<double> vec;
  typedef enum {
    SUBJECT_M   =0,
    PREDICATE_M =1,
    OBJECT_M    =2,
    SUBJECT_O   =3,
    PREDICATE_O =4,
    OBJECT_O    =5,
    UNKNOW      =-1
  }wordMode;
  struct sentence{
    std::string word,tag;
    wordMode mode;
    sentence(const sentence & s){
      word =s.word;
      tag  =s.tag;
    }
    sentence & operator=(const sentence & s){
      word =s.word;
      tag  =s.tag;
      return * this;
    }
    sentence(){}
  };
  class langsolv:word2vec {
    std::string mypath;
    NN::NN * net;
    std::mutex locker;
    public:
    double nearly;
    langsolv(
      const char * path,
      const char * a,
      const char * b,
      const char * c,
      const char * d,
      const char * e,
      const char * w,
      const char * l,
      const char * mpath
    ):word2vec(path,a,b,c,d,e,w,l){
      this->mypath=mpath;
      static int layer[]={ 60,480,2000,4000,4000,120 };
      FILE * f=fopen(mpath,"r");
      if(f==NULL){
        net=new NN::NN(0.25, 0.9, layer, 6, NN::SIGMOD);
      }else{
        net=new NN::NN(mpath);
        fclose(f);
        if(net->getInputSize()!=60){
          throw NNError();
        }
        if(net->getOutputSize()!=120){
          throw NNError();
        }
      }
    }
    ~langsolv(){
      if(net)delete net;
    }
    virtual void save(){
      locker.lock();
      net->save(mypath.c_str());
      locker.unlock();
    }
    virtual vec getVector(const std::string & wd){
      return (this->wordToVec(wd)/10000000.0d);
    }
    static inline int maxi(double * in){
      double v=in[0];
      int i=0;
      for(int j=1;j<6;j++){
        if(in[j]>v){
          i=j;
          v=in[j];
        }
      }
      return i;
    }
    static void set3arr(double * a,int i,const double * b){
      int ix=i*3;
      a[ix  ]=b[0];
      a[ix+1]=b[1];
      a[ix+2]=b[2];
    }
    static void set6arr(double * a,int i,const double * b){
      int ix=i*6;
      a[ix  ]=b[0];
      a[ix+1]=b[1];
      a[ix+2]=b[2];
      a[ix+3]=b[3];
      a[ix+4]=b[4];
      a[ix+5]=b[5];
    }
    static void get3arr(const double * a,int i,double * b){
      int ix=i*3;
      b[0]=a[ix  ];
      b[1]=a[ix+1];
      b[2]=a[ix+2];
    }
    static void get6arr(const double * a,int i,double * b){
      int ix=i*6;
      b[0]=a[ix  ];
      b[1]=a[ix+1];
      b[2]=a[ix+2];
      b[3]=a[ix+3];
      b[4]=a[ix+4];
      b[5]=a[ix+5];
    }
    virtual void solve(std::string word,std::list<sentence> & sen){
      if(word.empty())return;
      locker.lock();
      std::vector<std::string> words;
      this->CutForSearch(word,words);
      if(words.size()==0){
        locker.unlock();
        return;
      }
      
      int i=0;
      double input[60],output[120];
      double input_c[3],output_c[6];
      for(auto w:words){
        vec v=getVector(w);
        input_c[0]=v.x;
        input_c[1]=v.y;
        input_c[2]=v.z;
        set3arr(input,i,input_c);
        i++;
        if(i>20)break;
      }
      net->predict(input,output);
      
      for(int j=0;j<20;j++){
        get6arr(output,j,output_c);
        sentence s;
        s.word=words[i];
        s.mode=(wordMode)maxi(output);
        sen.push_back(s);
      }
      
      endf:
      locker.unlock();
    }
    virtual void train(const std::list<sentence> & sen){
      if(sen.size()==0)return;
      locker.lock();
      int i=0,m;
      double input[60], output[120];
      double input_c[3],output_c[6];
      for(auto w:sen){
        m=w.mode;
        for(int j=0;j<6;j++){
          output_c[j]=0;
        }
        if(m>5 || m<0){
        
        }else{
          output_c[m]=1;
        }
        set6arr(output,i,output_c);
        
        vec v=getVector(w.word);
        input_c[0]=v.x;
        input_c[1]=v.y;
        input_c[2]=v.z;
        set3arr(input,i,input_c);
        
        i++;
        if(i>20)break;
      }
      
      net->train(input,output);
      locker.unlock();
    }
  };
}
#endif