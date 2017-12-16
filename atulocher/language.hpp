#ifndef atulocher_language
#define atulocher_language
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
    word2vec * paser;
    std::map<std::string,std::string> keymeans;
    typedef std::vector<double> Vector;
    std::list<Vector> sent,keyword;
    std::vector<std::string> wds,kws;
    virtual void solve(const string & w){
      std::vector<cppjieba::KeywordExtractor::Word> kw;
      
      cutter->extractor.Extract(w,kw,5);
      for(auto it:kw){
        kws.push_back(it.word);
      }
      paser->sentToArr(kws,keyword,k);
      
      cutter->Cut(w,wds);
      paser->sentToArr(wds,sent,k);
      
      getkeymeans();
    }
    virtual void getkeymeans()=0;
  };
}
#endif