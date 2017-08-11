#ifndef atulocher_vord2vec
#define atulocher_vord2vec
#include "ksphere.hpp"
#include <list>
#include <vector>
#include <map>
#include "cppjieba/Jieba.hpp"
#include "language.hpp"
namespace atulocher{
  class wlist{
    RWMutex locker;
    std::map<std::string,double> lst;
    void readconfigline(const char * str){
      std::istringstream iss(str);
      double w;
      std::string s;
      iss>>s;
      iss>>w;
      if(s.empty())return;
      lst[s]=w;
    }
    void readconfig(const char * path){
      FILE * fp=NULL;
      fp=fopen(path,"r");
      if(fp==NULL)return;
      char buf[4096];
      while(!feof(fp)){
        bzero(buf,4096);
        fgets(buf,4096,fp);
        ksphere::confrep(buf);
        if(strlen(buf)<1)continue;
        readconfigline(buf);
      }
      fclose(fp);
    }
    void writeconfig(const char * str,double weig){
      char buf[4096];
      snprintf(buf,4096,"%s %f #time:%d\n",
        str,
        weig,
        time(0)
      );
      fwrite(buf,strlen(buf),1,fd);
    }
    FILE * fd;
    public:
    wlist(const char * path){
      readconfig(path);
      fd=fopen(path,"a");
    }
    ~wlist(){
      if(fd)fclose(fd);
    }
    void set(std::string kw,double w){
      locker.Wlock();
      lst[kw]=w;
      writeconfig(kw.c_str(),w);
      locker.unlock();
    }
    bool get(std::string kw,double * w){
      locker.Rlock();
      auto it=lst.find(kw);
      bool res;
      if(it==lst.end()){
        res=false;
      }else{
        res=true;
        *w=it->second;
      }
      locker.unlock();
      return res;
    }
  };
  class vord2vec:cppjieba::Jieba,wlist,langsolv{
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
    ksphere ks;
    vord2vec(
      const char * path,
      const char * a,
      const char * b,
      const char * c,
      const char * d,
      const char * e,
      const char * w,
      const char * l
    ):ks(path),Jieba(a,b,c,d,e),wlist(w),langsolv(l){
      
    }
    void learn(
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
    inline void setWeighter(std::string s,double w){
      //定义词语权重设置器
      //可用于定义否定词，语气词等
      //例：setWeighter("否",-1.0d);
      wlist::set(s,w);
    }
    octree::vec wordToVec(std::string &word){
      if(word.empty())return octree::vec(0,0,0);
      auto p=ks.find(word.c_str());
      if(p)return p->obj.position;  //有现成的，直接返回
      std::vector<std::string> words;
      this->Cut(word, words, true);
      double exp;
      if(words.size()==0)
        return octree::vec(0,0,0);  //无法分词
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
            double wr=1.0d;
            double wbuf;
            e=1.0d/(double)se.size();
            octree::vec S;
            for(auto seg:se){
              if(wlist::get(seg,&wbuf)){
                wr*=wbuf;
                continue;
              }
              auto kk=ks.find(seg);
              if(kk)
                S+=kk->obj.position*e;
            }
            ar.mean(S,exp*wr);
          }
        }
      }
      return ar.position;
    }
    void sentenceToVecs(
      std::string word,
      std::list< std::pair<std::string,octree::vec> > & res
    ){
      typedef std::pair<std::string,octree::vec> wp;
      std::vector<std::string> words;
      CutAll(word, words);
      for(auto it:words){
        auto v=wordToVec(it);
        octree::vec buf;
        if(v==octree::vec(0,0,0)){//无法解析
          if(numcheck(it,res)){   //尝试当作数字
          }else{
            memcpy(&buf,it.c_str(),sizeof(buf));//还是不能，强制转换
            res.push_back(wp(it,octree::vec(buf)));
          }
        }else{
          res.push_back(wp(it,v));
        }
      }
    }
  };
}
#endif