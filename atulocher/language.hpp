#ifndef atulocher_language
#define atulocher_language
#include <psyc/psyc.h>
#include <map>
#include <list>
#include <string>
#include <vector>
#include <set>
#include <mutex>
#include <exception>
#include "word2vec.hpp"
#include "sentree.hpp"
#include "dectree.hpp"
#include "active.hpp"
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
    
    PSNeuralNetwork                 * kmext,         //LSTM神经网络，抽取目标
                                    * tknn;          //LSTM encoder-decoder模型，为精确思维做预处理(直观思维)
    CRFPP::Tagger                   * senter,        //CRF模型，句法树构建
                                    * tagger;        //CRF模型，标注
    dectree::dectree                * dct;           //决策树，精确思维(逻辑思维)
    
    active                          * actives;
    lang(){
    }
    ~lang(){
    }
    virtual void solve(const string & w){
      if(tknn==NULL)return;
      if(dct==NULL)return;
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
      prethink();
      think();
    }
    virtual void addkm(int i){
      Vector bufv(512);
      auto pts=tree.allnode[i];
      std::string bufkm1,bufkm2;
      std::list<std::pair<std::string,std::string> > bufw;
      pts->foreach([](const sentree::node * n,void * arg){
        auto bufw=(std::list<std::pair<std::string,std::string> >*)arg;
        bufw->push_back(std::pair<std::string,std::string>(n->value,n->wordTag));
      },&bufw);
      actives->getkm(bufw,bufv,512,bufkm1,bufkm2);
      keymeans[bufkm1]=std::pair<Vector,std::string>(bufv,bufkm2);
    }
    virtual void addtg(int i){
      Vector bufv(512);
      auto pts=tree.allnode[i];
      std::string bufkm;
      std::list<std::string> bufw;
      pts->foreach([](const sentree::node * n,void * arg){
        auto bufw=(std::list<std::string>*)arg;
        bufw->push_back(n->value);
      },&bufw);
      actives->gettg(bufw,bufv,512,bufkm);
      target[bufkm]=bufv;
    }
    virtual void doactivity(double * arr,int l){
      std::string actname;
      actives->getActName(arr,l,actname);
      if(actname.empty())return;
      dct->doActivity(actname);
    }
    virtual void getkeymeans(){
      double buf[512],res[512];
      int i=0,
          times=sent.size();
      for(auto it:sent){
        int j;
        for(j=0;j<it.size();j++){
          if(j>=512)break;
          buf[j]=it[j];
        }
        PSPredict(kmext,buf,res,512,i,times);

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
    virtual void prethink(){
      int i=0;
      int encodetimes=keymeans.size()+target.size();
      int decodetimes=encodetimes*8;
      int times=encodetimes+decodetimes;
      double buf[516],res[516];
      //[512]:Normal
      //[513]:EOS_1
      //[514]:EOS_2
      //[515]:EOS_3
      //encode
      for(auto it:keymeans){
        set1(buf,515,512);
        Vector & vp=it.second.first;
        for(int j=0;j<512;j++)buf[j]=vp.at(j);
        PSPredict(tknn,buf,res,516,i,times);
        ++i;
      }
      //send EOS_1
      set1(buf,515,513);
      PSPredict(tknn,buf,res,516,i,times);
      ++i;
      //end
      for(auto it:target){
        Vector & vp=it.second;
        for(int j=0;j<512;j++)buf[j]=vp.at(j);
        PSPredict(tknn,buf,res,516,i,times);
        ++i;
      }
      //send EOS_2
      set1(buf,515,514);
      PSPredict(tknn,buf,res,516,i,times);
      ++i;
      //end
      //decode
      //receiving EOS_3
      for(int k=0;k<516;k++)res[k]=0;
      for(;i<times;i++){
        PSPredict(tknn,res,res,516,i,times);
        //if(getmax(res,516)==515)break;//EOS_3
        if(
          res[515]>res[514] &&
          res[515]>res[513] &&
          res[515]>res[512]
        )break;
        doactivity(res,512);
      }
    }
    virtual void think(){
      dct->compute();
    }
  };
}
#endif
