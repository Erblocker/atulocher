#ifndef atulocher_sentree
#define atulocher_sentree
#include <stdio.h>
#include <set>
#include <string>
#include <vector>
#include <exception>
#include "mempool.hpp"
namespace atulocher{
  using namespace std;
  class sentree_base{//构建依存句法树，便于使用crf
    public:
    typedef enum{
      NONE=0,
      ROOT=1
    }senmode;
    class ArrayError:public std::exception{};
    class TreeLenError:public std::exception{};
    class lk{
      public:
      int to;
      senmode mode;
      lk(){
        to=0;
        mode=NONE;
      }
      lk(const lk & l){
        to  =l.to;
        mode=l.mode;
      }
      lk(int t){
        to  =t;
        mode=NONE;
      }
      lk & operator=(const lk & l){
        to  =l.to;
        mode=l.mode;
        return *this;
      }
      lk & operator=(int t){
        to  =t;
        return *this;
      }
      bool operator>(const lk & l)const{
        return to>l.to;
      }
      bool operator<(const lk & l)const{
        return to<l.to;
      }
      bool operator==(const lk & l)const{
        return to==l.to;
      }
      bool operator!=(const lk & l)const{
        return to!=l.to;
      }
    };
    typedef set<int> linkset;
    class wd{
      public:
      string    word;
      linkset * belinked;
      lk        link;
      ~wd(){
        if(belinked)delete belinked;
      }
      wd(){
        belinked=NULL;
        link    =-1;
      }
      wd(const string w){
        belinked=NULL;
        link    =-1;
        word    =w;
      }
      wd(const wd & w){
        word     =w.word;
        belinked =w.belinked;
        link     =w.link;
      }
      wd & operator=(const wd & w){
        word     =w.word;
        belinked =w.belinked;
        link     =w.link;
        return *this;
      }
      void setB(lk p){
        if(belinked){
          belinked->insert(p.to);
        }else{
          belinked=new linkset;
          belinked->insert(p.to);
        }
      }
    };
    vector<wd> words;
    private:
    bool moveToRight(int p,int l){
      int len=words.size();
      if(p>=len || p<0){
        throw ArrayError();
        return false;
      }
      
      if(l<=0){
        throw TreeLenError();
        return false;
      }
      
      int i,t=0;
      
      //for(i=l-1;;i--){
      //  if(words[i].belinked)
      //    break;
      //  else
      //    t++;
      //}
      //if(t<l){
      //  throw TreeLenError;
      //  return false;
      //}
      
      for(i=len-1;i>=p;i--){
        moveLinkToRight(i,l);
      }
      return true;
    }
    void moveLinkToRight(int p,int ml){
      int len=words.size();
      wd & p1=words[p];
      wd & p2=words[((p+ml)>len-1)?(len-1):(p+ml)];
      if(!p1.belinked)
        return;
      else{
        if(!p2.belinked){
          p2.belinked=p1.belinked;
          p1.belinked=NULL;
          return;
        }else{
          for(auto it=p1.belinked->begin();it!=p1.belinked->end();){
            
            auto it2=it;
            p2.belinked->insert(*it);
            
            it++;
            
            p1.belinked->erase(it2);
          }
          return;
        }
      }
    }
    int getLeftElm(int p,int l){
      int len=words.size();
      if(p>=len || p<0){
        throw ArrayError();
        return 0;
      }
      
      if(l<=0){
        throw TreeLenError();
        return 0;
      }
      int i,pt=p;
      for(i=0;i<l;i++){
        begin:
          if(pt<=0)
            return 0;
          if(canLink(p,pt))
            continue;
          else{
            pt--;
            goto begin;
          }
      }
      return pt;
    }
    bool canLink(int p1,int p2){
      if(p1==p2)return false;
      int s=0,r=0;
      for(int i=p2;i<p1;i++){
        if(words[i].link!=-1)s++;
        if(words[i].belinked)r+=words[i].belinked->size();
      }
      if(s==r)
        return true;
      else
        return false;
    }
    bool createLink(int p,int l,senmode mode=NONE){
      int len=words.size();
      if(p>=len || p<0){
        throw ArrayError();
        return false;
      }
      if(l==0)return true;
      if(l>0){
        int np=((p+l)>len-1)?(len-1):(p+l);
        moveToRight(p+1,l);
        words[p ].link=np;
        words[p ].link.mode=mode;
        words[np].setB(p);
      }else{
        int np=getLeftElm(p,-l);
        words[p ].link=np;
        words[p ].link.mode=mode;
        words[np].setB(p);
      }
      return true;
    }
    public:
    virtual void getWord(string* ,int*,senmode*,int)=0;
    virtual int  getWordNum()=0;
    virtual void init(){
      int n=getWordNum();
      
      wd fst("");
      wd end("");
      
      
      string  owd;
      int     olk;
      senmode omd;
      
      words.push_back(fst);
      for(int i=0;i<n;i++){
        getWord(&owd,NULL,NULL,i);
        wd telm(owd);
        words.push_back(telm);
      }
      words.push_back(end);
      
      for(int i=0;i<n;i++){
        getWord(NULL,&olk,&omd,i);
        createLink(i+1,olk,omd);
      }
    }
  };
}
#endif