#ifndef atulocher_word2vec
#define atulocher_word2vec
#include <list>
#include <vector>
#include <map>
#include <fstream>
#include "utils.hpp"
#include "kbtree.hpp"
namespace atulocher{
  class word2vec{
    public:
    static size_t utf8_to_charset(const std::string &input,std::vector<std::string> &output){
      std::string ch; 
      for (register size_t i = 0, len = 0; i != input.length(); i += len) {
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
    virtual void sentToArr(const std::vector<std::string> & sent,std::list<std::vector<double> > & vs)=0;
    virtual void arrToSent(const std::list<std::vector<double> > & vs,std::vector<std::string> & sent)=0;
  };
  class word2vec_kbt:public word2vec{
    int fd;
    public:
    typedef kbtree::vec vec;
    struct wordnode{
      kbtree::value node;
      std::string word;
      wordnode * next;//gc
      void construct(){
        node.data=this;
      }
    };
    mempool<wordnode> pool;
    kbtree kbt;
    int k;
    std::map<std::string,wordnode*> words;
    word2vec_kbt(int kk,const char * path):kbt(kk){
      this->k=kk;
      readconf(path);
      fd=open(path,O_WRONLY|O_CREAT|O_APPEND,0644);
    }
    virtual void readconf(const char * path){
      std::ifstream fin(path);
      if(!fin.is_open())return;
      std::string wd;
      while(!fin.eof()){
        std::string method;
        fin>>method;
        if(method=="add"){
          
          vec posi(k);
          wd.clear();
          fin>>wd;
          for(int i=0;i<k;i++)
            fin>>posi[k];
          addWord(wd,posi);
          
        }else
        if(method=="del"){
          
          wd.clear();
          fin>>wd;
          removeWord(wd);
          
        }
      }
    }
    ~word2vec_kbt(){
      for(auto it:words){
        if(it.second)pool.del(it.second);
      }
      if(fd!=-1)close(fd);
    }
    inline void setzero(vec & v){
      for(int i=0;i<v.size();i++){
        v[i]=0;
      }
    }
    virtual void wconf_add(const std::string & wd,const vec & posi){
      char buf[512];
      snprintf(buf,512,"add %s ",wd.c_str());
      write(fd,buf,strlen(buf));
      for(int i=0;i<k;i++){
        bzero(buf,512);
        snprintf(buf,512,"%f ",posi.at(k));
        write(fd,buf,strlen(buf));
      }
    }
    virtual void wconf_del(const std::string & wd){
      char buf[512];
      snprintf(buf,512,"del %s ",wd.c_str());
      write(fd,buf,strlen(buf));
    }
    inline void addvec(vec & out,const vec & in,double w)const{
      for(int i=0;i<k;i++){
        out[i]+=in[k]*w;
      }
    }
    virtual void sentToArr(const std::vector<std::string> & sent,std::list<std::vector<double> > & vs){
      for(auto it:sent){
        vec tmp(k);
        wordToVec(it,tmp);
        vs.push_back(tmp);
      }
    }
    virtual void arrToSent(const std::list<vec> & vs,std::vector<std::string> & sent){
      for(auto it:vs){
        std::string thwd;
        vecToWord(it,thwd);
        sent.push_back(thwd);
      }
    }
    virtual void learn(
      const std::string & word,
      const std::list< std::pair<std::string,double> > & mean
    ){
      vec posi(k);
      setzero(posi);
      for(auto it:mean){
        auto wit=words.find(it.first);
        if(wit==words.end())continue;
        if(!(wit->second))continue;
        addvec(posi,wit->second->node.position,it.second);
      }
      setWord(word,posi);
    }
    virtual bool removeWord(const std::string & word){
      auto it=this->words.find(word);
      if(it==this->words.end())return false;
      if(it->second){
        kbt.erase(&(it->second->node));
        pool.del(it->second);
      }
      words.erase(it);
      return true;
    }
    inline void delWord(const std::string & word){
      if(removeWord(word))
        wconf_del(word);
    }
    inline void setWord(const std::string & word,const vec & mean){
      if(addWord(word,mean))
        wconf_add(word,mean);
    }
    virtual bool addWord(const std::string & word,const vec & mean){
      auto it=this->words.find(word);
      if(it!=this->words.end())return false;
      auto pp=pool.get();
      pp->construct();
      pp->word=word;
      words[word]=pp;
      pp->node.position=mean;
      kbt.insert(&(pp->node));
      return true;
    }
    void vecToWord(const vec & v,std::string & wd,double R=0.3){
      vec beg(k),end(k);
      for(int i=0;i<k;i++){
        beg[k]=v[k]-R;
        end[k]=v[k]+R;
      }
      wordnode * wp;
      kbt.find([](kbtree::value * v,void * arg){
        auto wp=(wordnode**)arg;
        *wp=(wordnode*)v->data;
      },beg,end,&wp,1);
      wd=wp->word;
    }
    void wordToVec(const std::string & wd,vec & out){
      auto it=words.find(wd);
      if(it!=words.end()){
        out=it->second->node.position;
        return;
      }
      std::vector<std::string> wds;
      this->utf8_to_charset(wd, wds);
      setzero(out);
      if(wds.size()==0)return;
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
      }pp(wds.size());
      for(auto wit:wds){
        pp.next();
        auto sit=words.find(wit);
        if(sit!=words.end())
          addvec(out,sit->second->node.position,pp.w);
      }
    }
  };
}
#endif
