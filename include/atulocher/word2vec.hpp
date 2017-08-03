#ifndef atulocher_vord2vec
#define atulocher_vord2vec
#include "ksphere.hpp"
#include <list>
#include <vector>
#include "cppjieba/Jieba.hpp"
namespace atulocher{
  class vord2vec:cppjieba::Jieba{
    //汉字转向量
    //使用前请自己准备cppjieba字典
    //还有足够的数据来训练思维球
    //我这里可没有训练好的思维球模型啊（那玩意儿我没有(>_<)）
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
    bool numcheck(const std::string &word,std::list<octree::vec> & res){
      std::string w=word;
      if(!cleanNum(w))return false;
      std::istringstream iss(w);
      octree::vec vv;
      //for(nit i=0;i<4;i++){
        iss>>vv.x;
        iss>>vv.y;
        iss>>vv.z;
      //}
      res.push_back(vv);
    }
    public:
    ksphere ks;
    vord2vec(
      const char * path,
      const char * a,
      const char * b,
      const char * c,
      const char * d,
      const char * e
    ):ks(path),Jieba(a,b,c,d,e){
      
    }
    void learn(
      const std::string & word,
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
    octree::vec wordToVec(const std::string &word){
      if(word.empty())return octree::vec(0,0,0);
      auto p=ks.find(word.c_str());
      if(p)return p->obj.position;  //有现成的，直接返回
      std::vector<std::string> words;
      this->Cut(word, words, true);
      double exp;
      if(words.size()==0)
        return octree::vec(0,0,0);
      else
        exp=1.0d/(double)words.size();
      ksphere::adder ar(&ks);
      ar.readonly=true;
      for(auto s:words){
        if(ar.mean(s,exp)){
          //有，直接加入
        }else{
          //否则,尝试推测词语意思
          std::vector<std::string> se;
          CutForSearch(s,se);
          double e;
          if(se.size()==0)
            continue;
          else{
            e=1.0d/(double)se.size();
            octree::vec S;
            for(auto seg:se){
              auto kk=ks.find(seg);
              if(kk)
                S+=kk->obj.position*e;
            }
            ar.mean(S,exp);
          }
        }
      }
    }
    void sentenceToVecs(
      const std::string &word,
      std::list<octree::vec> & res
    ){
      std::vector<std::string> words;
      CutAll(word, words);
      for(auto it:words){
        auto v=wordToVec(it);
        octree::vec buf;
        if(v==octree::vec(0,0,0)){
          if(numcheck(it,res)){
          }else{
            memcpy(&buf,it.c_str(),sizeof(buf));
            res.push_back(octree::vec(buf));
          }
        }else{
          res.push_back(v);
        }
      }
    }
  };
}
#endif