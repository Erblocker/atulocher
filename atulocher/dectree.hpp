#ifndef atulocher_dectree
#define atulocher_dectree
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <list>
#include <map>
#include <set>
#include <vector>
#include <sstream>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include <unistd.h>
#include <lua.hpp>
#include "mempool.hpp"
namespace atulocher{
  namespace dectree{
    using namespace std;
    //匹配前缀
    //s1:前缀
    //s2:字符串
    bool prefix_match(const char * s1,const char * s2){
      const char * sp=s1;
      const char * p2=s2;
      while(*sp){
       if((*sp)!=(*p2))return 0;
       sp++;
       p2++;
      }
      return 1;
    }
    struct prob{
      int all;//总次数
      int num;//发生概率
      inline double getProb(){
        return (double)num/(double)all;
      }
      void loadString(const string & str){
        istringstream iss(str);
        iss>>all;
        iss>>num;
      }
      void toString(string & str){
        char buf[128];
        snprintf(buf,128,"%d %d",all,num);
        str=buf;
      }
    };
    class probinfo{
      public:
      int times;
      string key;
      string actname;
      bool operator<(const probinfo & p)const{
        return times>p.times;
      }
      bool operator>(const probinfo & p)const{
        return times<p.times;
      }
      bool operator==(const probinfo & p)const{
        return times==p.times;
      }
    };
    class info:public probinfo{
      public:
      info * next,
           * gc_next;
      set<info*> depend;
      string name;
      string answer;
      string res;
    };
    typedef mempool_auto<info> GC;
    class base{
      public:
      GC                gc;
      leveldb::DB     * db;
      map<string,info*> known;
      set<string>       target;
      string            treename;
      int        depth,       //搜索深度
                 step,        //行动次数
                 nowDepth,    //当前深度
                 lastDepth,   //上次获得结果的深度
                 testTimes,   //尝试次数
                 testDepth,   //尝试层数
                 itTimes;     //迭代次数
      bool       succeed;
      virtual void findEvent(const list<string> &kl)=0;
      virtual void print(void(*callback)(const char*,void*),void * arg)=0;
      virtual void doActivity(const string & actname)=0;
      
      virtual void learnOne(const list<string> &kl){
        string k;
        getKey(kl,k);
        if(!logEvent(k))findEvent(kl);
      }
      virtual void learn(const list<string> &path){
        auto last=path.end();
        for(auto here=path.begin();here!=path.end();here++){
          for(auto it:known){
            for(auto targ:target){
              list<string> kl;
              kl.push_back(it.first);
              kl.push_back(targ);
              if(last!=path.end())
                kl.push_back(*last);
              else
                kl.push_back(" ");
              kl.push_back(*here);
              learnOne(kl);
            }
          }
          last=here;
        }
      }
      virtual void getKey(const list<string> &dep,string & name){
        name="dectree_prob_";
        for(auto s:dep){
          name+=treename+"_";
          name+=s;
        }
      }
      virtual info * getknown(string name){
        auto d=known.find(name);
        if(d!=known.end())
          return d->second;
        else
          return NULL;
      }
      virtual void addknown(string name,string answer,string res,const set<string> & depend){
        auto p=gc.get();
        p->name=name;
        p->answer=answer;
        p->res =res;
        for(auto it:depend){
          auto d=known.find(it);
          if(d!=known.end()){
            p->depend.insert(d->second);
          }
        }
        known[name]=p;
      }
      virtual bool logEvent(const string & k,int atms=1){
        string v;
        if(
          !db->Get(
            leveldb::ReadOptions(),
            k,
            &v
          ).ok()
        )return false;
        if(v.empty())return false;
        int times;
        string actname;
        istringstream iss(v);
        iss>>times;
        iss>>actname;
        if(actname.empty())return false;
        times+=atms;
        char tm[32];
        snprintf(tm,32,"%d",times);
        v=tm;
        v+=" ";
        v+=actname;
        db->Put(
          leveldb::WriteOptions(),k,v
        );
      }
      virtual void setEvent(
        const string & keyword,
        const string & parname,
        const string & actname,
        const string & targ
      ){
        list<string> nl;
        nl.push_back(keyword);
        nl.push_back(targ);
        nl.push_back(parname);
        string k;
        getKey(nl,k);
        char tm[32];
        snprintf(tm,32,"%d",time(NULL));
        db->Put(
          leveldb::WriteOptions(),k,tm
        );
        nl.push_back(actname);
        getKey(nl,k);
        db->Put(
          leveldb::WriteOptions(),k,string("1 ")+actname
        );
      }
      virtual bool check(){
        for(auto it:target)
          if(known.find(it)==known.end())
            return false;
        succeed=true;
        return true;
      }
      virtual void search(const list<string> & dep,map<string,int> & acts){
        string k;
        getKey(dep,k);
        auto ks=k.c_str();
        leveldb::ReadOptions options;
        //options.snapshot = db->GetSnapshot();
        leveldb::Iterator* it = db->NewIterator(options);
        set<probinfo> pis;
        for(it->Seek(ks);(it->Valid() && prefix_match(ks,it->key().data()));it->Next()){
          probinfo p;
          p.key=it->key().data();
          istringstream iss(it->value().data());
          iss>>p.times;
          iss>>p.actname;
          if(p.actname.empty())continue;
          pis.insert(p);
        }
        int i=0;
        for(auto pi:pis){
          if(succeed)break;
          acts[pi.actname]+=pi.times;
          i++;
          if(i>testTimes)break;
        }
      }
      virtual bool compute(){
        this->lastDepth=0;
        string begact=" ";
        for(int i=0;i<itTimes;i++){
          if(succeed)return true;
          searchAll(begact);
        }
        return false;
      }
      virtual void searchAll(const string & actname){
        step--;
        if(step<0)return;
        nowDepth++;
        if(nowDepth>depth){
          nowDepth--;
          return;
        }
        if((nowDepth-lastDepth)>testDepth){
          nowDepth--;
          return;
        }
        map<string,int> acts;
        set<probinfo>   actl;
        for(auto it:known){
          for(auto targ:target){
            list<string> kl;
            kl.push_back(it.first);
            kl.push_back(targ);
            kl.push_back(actname);
            search(kl,acts);
          }
        }
        for(auto it:acts){
          probinfo p;
          p.actname=it.first;
          p.times=it.second;
          actl.insert(p);
        }
        acts.clear();
        int i=0;
        for(auto it:actl){
          check();
          if(succeed)return;
          doActivity(it.actname);
          i++;
          if(i>testTimes)break;
        }
        nowDepth--;
      }
      base(){
        succeed=false;
      }
      ~base(){}
    };
    class activity:public base{
      public:
      map<string,string> env;
      lua_State * L;
      activity(){
        L=luaL_newstate();
        this->luaopen();
      }
      ~activity(){
        lua_close(L);
      }
      static inline int isptr(lua_State * L,int p){
        return (lua_isuserdata(L,p));
      }
      static inline void * toptr(lua_State * L,int p){
        if(lua_isuserdata(L,p)){
          auto pt=(void **)lua_touserdata(L,p);
          return *pt;
        }else{
          return NULL;
        }
      }
      static inline void pushptr(lua_State * L,void * ptr){
        auto pt=(void **)lua_newuserdata(L,sizeof(void*));
        *pt=ptr;
      }
      
      #define GETSELF \
        if(!isptr(L,1))return 0;\
        auto self=(activity*)toptr(L,1); 
      
      static int lua_getknown(lua_State * L){
        GETSELF;
        if(!lua_isstring(L,2))return 0;
        auto it=self->known.find(lua_tostring(L,2));
        if(it==self->known.end())return 0;
        
        auto pinfo=it->second;
        
        lua_createtable(L,0,4);
        
        lua_pushstring(L,"name");
        lua_pushstring(L,pinfo->name.c_str());
        lua_settable(L,-3);
        
        lua_pushstring(L,"answer");
        lua_pushstring(L,pinfo->answer.c_str());
        lua_settable(L,-3);
        
        lua_pushstring(L,"result");
        lua_pushstring(L,pinfo->res.c_str());
        lua_settable(L,-3);
        
        lua_pushstring(L,"depend");
        lua_createtable(L,pinfo->depend.size(),0);//create new array
        lua_pushnil(L);
        lua_rawseti(L,-2,0);  //fill array[i][0]
        int i=1;
        for(auto itd:pinfo->depend){
          lua_pushstring(L,itd->name.c_str());
          lua_rawseti(L,-2,i);
          i++;
        }
        lua_settable(L,-3);
        
        return 1;
      }
      static int lua_getenv(lua_State * L){
        GETSELF;
        if(!lua_isstring(L,2))return 0;
        auto it=self->env.find(lua_tostring(L,2));
        if(it==self->env.end())return 0;
        lua_pushstring(L,it->second.c_str());
        return 1;
      }
      static int lua_callactivity(lua_State * L){
        GETSELF;
        if(!lua_isstring(L,2))return 0;
        self->callactivity(lua_tostring(L,2));
        return 0;
      }
      static int lua_addknown(lua_State * L){
        GETSELF;
        if(!lua_isstring(L,2))return 0;
        if(!lua_isstring(L,3))return 0;
        if(!lua_isstring(L,4))return 0;
        set<string> dep;
        
        string a=lua_tostring(L,2),
               b=lua_tostring(L,3),
               c=lua_tostring(L,4);
              
        if(lua_istable(L,-1)){
          int n = luaL_len(L,-1);
          for(int i=1;i<=n;i++){
            lua_pushnumber(L,i);
            lua_gettable(L,-2);
            if(!lua_isstring(L,-1))continue;
            dep.insert(lua_tostring(L,-1));
            lua_pop(L,-1);
          }
        }
        self->addknown(a,b,c,dep);
        lua_pushboolean(L,1);
        return 1;
      }
      
      void luaopen(){
        static luaL_Reg reg[]={
          {"addknown",lua_addknown},
          {"getknown",lua_getknown},
          {"getenv",  lua_getenv},
          {"callactivity",lua_callactivity},
          NULL,NULL
        };
        luaL_openlibs(L);
        lua_newtable(L);
        luaL_setfuncs(L, reg, 0);
        lua_setglobal(L,"ST");
      }
      virtual void callactivity(const char * name){
        auto lt=lua_newthread(L);
        
        luaL_loadfile(lt,name);
        
        lua_getglobal(lt,"main");
        if(!lua_isfunction(lt,-1))return;
        
        auto pt=(void **)lua_newuserdata(lt,sizeof(void*));
        *pt=this;
        
        if(lua_pcall(lt,1,1,0)!=0){
          
        }else
          if(lua_isinteger(lt,-1))
            if(lua_tointeger(lt,-1)==0)
              lastDepth=nowDepth;
        lua_pop(lt,1);
      }
      virtual void doActivity(const string & actname){
        string k="lua_activity_";
        k+=actname;
        string v;
        if(
          !db->Get(
            leveldb::ReadOptions(),
            k,
            &v
          ).ok()
        )return;
        if(v.empty())return;
        callactivity(v.c_str());
      }
      #undef GETSELF
    };
  }
}
#endif