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
    double   weight;
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
    NN::RNN * net;
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
      static int layer[]={ 3,480,2000,4000,4000,7 };
      FILE * f=fopen(mpath,"r");
      if(f==NULL){
        net=new NN::RNN(0.25, 0.9, layer, 6, NN::SIGMOD);
      }else{
        net=new NN::RNN(mpath);
        fclose(f);
        if(net->getInputSize()!=3){
          throw NNError();
        }
        if(net->getOutputSize()!=7){
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
    static inline vec getVector(const vec & in){
      return (in/10000000.0d);
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
    virtual void getKeyword(std::string word,std::list<vec> & kw){
      if(word.empty())return;
      locker.lock();
      std::vector<cppjieba::KeywordExtractor::Word>           keywordres;
      this->extractor.Extract(word,keywordres,5);
      for(auto it:keywordres){
        kw.push_back(this->getVector(it.word)*it.weight);
      }
      locker.unlock();
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
      double input[3],output[7];
      for(auto w:words){
        vec v=getVector(w);
        input[0]=v.x;
        input[1]=v.y;
        input[2]=v.z;
        net->predict(input,output,i==0);
        sentence s;
        s.word=words[i];
        s.mode=(wordMode)maxi(output);
        s.weight=output[6];
        sen.push_back(s);
        i++;
      }
      
      endf:
      locker.unlock();
    }
    virtual void train(const std::list<sentence> & sen){
      if(sen.size()==0)return;
      locker.lock();
      int i=0,m;
      double input[3], output[7];
      for(auto w:sen){
        m=w.mode;
        for(int j=0;j<6;j++){
          output[j]=0;
        }
        if(m>5 || m<0){
        
        }else{
          output[m]=1;
        }
        output[6]=w.weight;
        vec v=getVector(w.word);
        input[0]=v.x;
        input[1]=v.y;
        input[2]=v.z;
        net->train(input,output,i==0);
        i++;
      }
      
      locker.unlock();
    }
  };
}
#endif