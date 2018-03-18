#ifndef atulocher_active
#define atulocher_active
#include <crfpp.h>
#include <sstream>
#include "language.hpp"
#include "rpc.hpp"
#include "actscript.hpp"
namespace atulocher{
  class active:public lang{
    public:
    CRFPP::Tagger * tagger;
    int fd;
    void remote_init(){
      RakNet::BitStream res,ret;
      res<<fd;
      rpc.call("graph_create",&res,&ret);
    }
    void remote_destroy(){
      RakNet::BitStream res,ret;
      res<<fd;
      rpc.call("graph_destroy",&res,&ret);
    }
    void remote_add(const std::list<std::string> & wds){
      RakNet::BitStream res,ret;
      std::string resstr;
      char fdstr[32];
      
      snprintf(fdstr,32,"%d ",fd);
      resstr=fdstr;
      for(auto it:wds)resstr+=(it+" ");
      
      res.WriteCompressed(resstr.c_str());
      rpc.call("graph_add",&res,&ret);
    }
    void remote_find(const std::set<int> & cond){
      RakNet::BitStream res,ret;
      res<<fd;
      for(auto it:cond)res<<it;
      rpc.call("graph_find",&res,&ret);
    }
    
    void getObj(
      const std::list<std::string> & wds,
      std::vector<double> & arr,
      int len,
      std::string & key
      ){
      RakNet::BitStream res,ret;
      char hd[128];
      
      snprintf(hd,128,"%d %d ",fd,len);
      std::string resstr=hd;
      
      for(auto it:wds)resstr+=(it+" ");
      
      res.WriteCompressed(resstr.c_str());
      rpc.call("objs_getobj",&res,&ret);
      
      arr.resize(len);
      for(int i=0;i<len;i++)ret>>arr[i];
      ret>>key;
    }
    
    virtual void getkm(
      const std::list<std::pair<std::string,std::string> > & kms,
      std::vector<double> & arr,
      int len,
      std::string & key,
      std::vector<std::string> & value){
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
      std::list<std::string> objs;
      std::vector<std::string> tags;
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
      value=tags;
    }
    virtual void gettg(
      const std::list<std::string> & kms,
      std::vector<double> & arr,
      int len,
      std::string & key){
      getObj(kms,arr,len,key);
    }
    
    virtual void doactivity(double * arr,int l){
      
    }
  };
}
#endif
