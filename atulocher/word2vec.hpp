#ifndef atulocher_word2vec
#define atulocher_word2vec
#include "ksphere.hpp"
#include <list>
#include <vector>
#include <map>
namespace atulocher{
  class word2vec_base{
    //汉字转向量
    //使用前请自己准备足够的数据来训练ksphere
    //我这里可没有训练好的模型啊（那玩意儿我没有(>_<)）
    bool cleanNum(std::string & wd){
      int i;
      auto s=wd.c_str();
      bool r=false;
      for(i=0;s[i]!='\0';i++){
        if(s[i]=='.')continue;
        if(isdigit(s[i])){
          r=true;
          continue;
        }
        wd[i]=' ';
      }
      return r;
    }
    bool numcheck(const std::string &word,std::list< std::pair<std::string,octree::vec> > & res){
      typedef std::pair<std::string,octree::vec> wp;
      std::string w=word;
      if(!cleanNum(w))return false;
      std::istringstream iss(w);
      octree::vec vv;
      //for(nit i=0;i<4;i++){
        iss>>vv.x;
        iss>>vv.y;
        iss>>vv.z;
      //}
      res.push_back(wp(word,vv));
    }
    public:
    typedef octree::vec vec;
    ksphere ks;
    word2vec_base(const char * path):ks(path){
      
    }
    static size_t utf8_to_charset(const std::string &input,std::vector<std::string> &output){
      std::string ch; 
      for (size_t i = 0, len = 0; i != input.length(); i += len) {
        unsigned char byte = (unsigned)input[i];
        if (byte >= 0xFC) // lenght 6
          len = 6;  
        else if (byte >= 0xF8)
          len = 5;
        else if (byte >= 0xF0)
         len = 4;
        else if (byte >= 0xE0)
         len = 3;
        else if (byte >= 0xC0)
         len = 2;
        else
         len = 1;
        ch = input.substr(i, len);
        output.push_back(ch);
      }
      return output.size();
    }
    virtual void learn(
      std::string & word,
      const std::list< std::pair<std::string,double> > & mean
    ){
      ksphere::adder ar(&ks);
      std::string v("wordmeans:");
      char buf[256];
      for(auto it:mean){
        ar.mean(it.first,it.second);
        snprintf(buf,256,"%s:%f;",it.first.c_str(),it.second);
        v+=buf;
      }
      char sbuf[3500];
      memcpy(sbuf,v.c_str(),3500);
      ar.add(word,sbuf);
    }
    virtual octree::vec wordToVec(std::string word,double * arr=NULL,int arrs=0){
      if(word.empty())return octree::vec(0,0,0);
      auto p=ks.find(word.c_str());
      if(p){
        if(arr){
          ks.toArray(arr,arrs,p);
        }
        return p->obj.position;  //有现成的，直接返回
      }
      std::vector<std::string> words;
      this->utf8_to_charset(word, words);
      double exp;
      if(words.size()==0)
        return octree::vec(0,0,0);
      else
        exp=1.0d/(double)words.size();
      class PP{
        double step;
        public:
        double w;
        PP(int l){
          double s=(l*(s-1))/2.0d;
          step=1.0d/s;
          w=0;
        }
        void next(){
          w+=step;//权重逐渐增大
        }
      }pp(words.size());
      ksphere::adder ar(&ks);
      ar.readonly=true;
      for(auto s:words){
        pp.next();
        if(ar.mean(s,exp*pp.w)){
          //有，直接加入
          if(arr){
            ks.toArray(arr,arrs,s,exp*pp.w);
          }
        }
      }
      return ar.position;
    }
    virtual void getSimiler(const std::string & str,void(*callback)(const std::string&,const vec&,void*),double range,void *arg){
      auto p=wordToVec(str);
      
      if(!callback)return;
      struct self_o{
        void(*callback)(const std::string&,const vec&,void*);
        void * arg;
      }self;
      self.arg=arg;
      self.callback=callback;
      
      ks.getnear(p,[](ksphere::knowledge * ks,void * s){
        auto self=(self_o*)s;
        self->callback(ks->key,ks->obj.position,self->arg);
      },range,&self);
    }
  };
  class word2vec:public word2vec_base{
    
  };
}
#endif