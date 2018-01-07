#ifndef atulocher_language
#define atulocher_language
#ifndef atulocher_language_tensor_size
  #define atulocher_language_tensor_size 512
#endif
#include "ann.hpp"
#include <map>
#include <list>
#include <string>
#include <vector>
#include <set>
#include <mutex>
#include <exception>
#include "word2vec.hpp"
#include "sentree.hpp"
#include <crfpp.h>
#include "cppjieba/Jieba.hpp"
namespace atulocher{
  class lang{
    public:
    int k;
    cppjieba::Jieba * cutter;
    typedef std::vector<double> Vector;
    word2vec * paser,
             * conver;
    std::map<
      std::string,std::pair<Vector,std::string>
    >                                 keymeans;      //已知
    std::map<std::string,Vector>      target;        //求解
    std::list<Vector>                 sent,          //句子的词语  (vector)
                                      keyword;       //关键词      (vector)
    std::vector<std::string>          wds,           //句子的词语
                                      kws;           //关键词
    sentree                           tree;          //句法树
    
    ann::FD                           kmext,         //LSTM神经网络，抽取目标
                                      tknn;          //LSTM encoder-decoder模型，思维
    
    CRFPP::Tagger                   * senter,        //CRF模型，句法树构建
                                    * tagger;        //CRF模型，标注
    lang(){
    }
    ~lang(){
    }
    virtual void solve(const string & w){
      if(tknn==0)return;
      std::vector<cppjieba::KeywordExtractor::Word> kw;
      
      cutter->extractor.Extract(w,kw,5);
      for(auto it:kw){
        kws.push_back(it.word);
      }
      paser->sentToArr(kws,keyword);
      
      cutter->Cut(w,wds);
      paser->sentToArr(wds,sent);
      
      tree.buildTag (wds,this->tagger);
      tree.buildTree(this->senter);
      
      getkeymeans();
      think();
    }
    virtual void addkm(int i){
      Vector bufv(atulocher_language_tensor_size);
      auto pts=tree.allnode[i];
      std::string bufkm1,bufkm2;
      std::list<std::pair<std::string,std::string> > bufw;
      pts->foreach([](const sentree::node * n,void * arg){
        auto bufw=(std::list<std::pair<std::string,std::string> >*)arg;
        bufw->push_back(std::pair<std::string,std::string>(n->value,n->wordTag));
      },&bufw);
      this->getkm(bufw,bufv,atulocher_language_tensor_size,bufkm1,bufkm2);
      keymeans[bufkm1]=std::pair<Vector,std::string>(bufv,bufkm2);
    }
    virtual void addtg(int i){
      Vector bufv(atulocher_language_tensor_size);
      auto pts=tree.allnode[i];
      std::string bufkm;
      std::list<std::string> bufw;
      pts->foreach([](const sentree::node * n,void * arg){
        auto bufw=(std::list<std::string>*)arg;
        bufw->push_back(n->value);
      },&bufw);
      this->gettg(bufw,bufv,atulocher_language_tensor_size,bufkm);
      target[bufkm]=bufv;
    }
    virtual void getkeymeans(){
      double buf[atulocher_language_tensor_size],res[atulocher_language_tensor_size];
      int i=0,
          times=sent.size();
      for(auto it:sent){
        int j;
        for(j=0;j<it.size();j++){
          if(j>=atulocher_language_tensor_size)break;
          buf[j]=it[j];
        }
        ann::Predict(kmext,buf,res,atulocher_language_tensor_size,i,times);

        int max_idx=0;
        double max=0;
        for(j=0;j<3;j++){
          double a=res[j];
          if(a > max) {
            max = a;
            max_idx = j;
          }
        }
        switch(max_idx){
          case 1:
            addkm(i);
          break;
          case 2:
            addtg(i);
          break;
        }
        i++;
      }
    }
    static inline void set1(
      double * arr,
      unsigned int len,
      unsigned int i
    ){
      for(int j=0;j<len;j++)arr[j]=0;
      if(i>=len)return;
      arr[i]=1;
    }
    static inline int getmax(double * arr,int len){
      int max_idx=0;
      double max=0;
      for(int j=0;j<len;j++){
        double a=arr[j];
        if(a > max) {
          max = a;
          max_idx = j;
        }
      }
      return max_idx;
    }
    virtual void think(){
      int i=0;
      int encodetimes=keymeans.size()+target.size();
      int decodetimes=encodetimes*8;
      int times=encodetimes+decodetimes;
      double buf[atulocher_language_tensor_size+4],res[atulocher_language_tensor_size+4];
      //[atulocher_language_tensor_size+0]:Normal
      //[atulocher_language_tensor_size+1]:EOS_1
      //[atulocher_language_tensor_size+2]:EOS_2
      //[atulocher_language_tensor_size+3]:EOS_3
      //encode
      for(auto it:keymeans){
        set1(buf,atulocher_language_tensor_size+3,atulocher_language_tensor_size);
        Vector & vp=it.second.first;
        for(int j=0;j<atulocher_language_tensor_size;j++)buf[j]=vp.at(j);
        ann::Predict(tknn,buf,res,atulocher_language_tensor_size+4,i,times);
        ++i;
      }
      //send EOS_1
      set1(buf,atulocher_language_tensor_size+4,atulocher_language_tensor_size+1);
      ann::Predict(tknn,buf,res,atulocher_language_tensor_size+4,i,times);
      ++i;
      //end
      for(auto it:target){
        Vector & vp=it.second;
        for(int j=0;j<atulocher_language_tensor_size;j++)buf[j]=vp.at(j);
        ann::Predict(tknn,buf,res,atulocher_language_tensor_size+4,i,times);
        ++i;
      }
      //send EOS_2
      set1(buf,atulocher_language_tensor_size+4,atulocher_language_tensor_size+2);
      ann::Predict(tknn,buf,res,atulocher_language_tensor_size+4,i,times);
      ++i;
      //end
      //decode
      //receiving EOS_3
      for(int k=0;k<atulocher_language_tensor_size+4;k++)res[k]=0;
      for(;i<times;i++){
        ann::Predict(tknn,res,res,atulocher_language_tensor_size+4,i,times);
        //if(getmax(res,atulocher_language_tensor_size+4)==atulocher_language_tensor_size+3)break;//EOS_3
        if(
          res[atulocher_language_tensor_size+3]>res[atulocher_language_tensor_size+2] &&
          res[atulocher_language_tensor_size+3]>res[atulocher_language_tensor_size+1] &&
          res[atulocher_language_tensor_size+3]>res[atulocher_language_tensor_size]
        )break;
        doactivity(res,atulocher_language_tensor_size);
      }
    }
    virtual void doactivity(double * arr,int l)=0;
    virtual void getkm(
      const std::list<std::pair<std::string,std::string> > & kms,
      std::vector<double> & arr,
      int len,
      std::string & key,
      std::string & value)=0;
    virtual void gettg(
      const std::list<std::string> & kms,
      std::vector<double> & arr,
      int len,
      std::string & key)=0;
  };
}
#endif
