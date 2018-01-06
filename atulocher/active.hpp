#ifndef atulocher_active
#define atulocher_active
#include <crfpp.h>
#include "dectree.hpp"
namespace atulocher{
  class active{
    public:
    CRFPP::Tagger * tagger;
    void getActName(const double * arr,int len,std::string & name){
      
    }
    void getObj(
      const std::list<std::string> & wds,
      std::vector<double> & arr,
      int len,
      std::string & key
      ){
    }
    void getVal(
      const std::list<std::string> & wds,
      std::string & key){
      
    }
    void getkm(
      const std::list<std::pair<std::string,std::string> > & kms,
      std::vector<double> & arr,
      int len,
      std::string & key,
      std::string & value){
      char buf[1024];
      if(tagger==NULL)return;
      tagger->clear();
      for(auto it:kms){
        snprintf(buf,1024,"%s %s",
          it.first .c_str(),
          it.second.c_str()
        );
        tagger->add(buf);
      }
      if(!tagger->parse())return;
      std::list<std::string> objs,tags;
      auto it=kms.begin();
      std::string predict;
      for (size_t i = 0; i < tagger->size(); ++i) {
        if(it==kms.end())break;
        
        predict=tagger->y2(i);
        if(predict=="o"){
          objs.push_back(it->first);
        }else
        if(predict=="t"){
          tags.push_back(it->first);
        }
        
        it++;
      }
      getObj(objs,arr,len,key);
      getVal(tags,value);
    }
    void gettg(
      const std::list<std::string> & kms,
      std::vector<double> & arr,
      int len,
      std::string & key){
      getObj(kms,arr,len,key);
    }
  };
}
#endif
