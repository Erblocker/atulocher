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
#include "utils.hpp"
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
    class info{
      public:
      info * next,
           * gc_next;
      set<info*> depend;
      string name;
      string answer;
      string res;
    };
    typedef mempool_auto<info> GC;
    struct keyname{
      string keyword;
      string parname;
      string actname;
      string targ;
    };
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
                 searchTimes, //搜索次数
                 itTimes;     //迭代次数
      bool       succeed;
      virtual void findEvent(const keyname &kl)=0;
      virtual void doActivity(const string & actname)=0;
      virtual void print(list<string> & res,info * t,int maxsearchdeep=32){
        if(maxsearchdeep<0)return;
        res.push_front(t->answer);
        for(auto it:t->depend)
          this->print(res,it,maxsearchdeep-1);
      }
      virtual void print(list<string> & res){
        for(auto it:target){
          auto p=known.find(it);
          if(p!=known.end()){
            this->print(res,p->second);
          }
        }
        
      }
      virtual void learnOne(const keyname &kl){
        string k;
        getKey(kl,k);
        if(!logEvent(k))findEvent(kl);
      }
      virtual void learn(const list<string> &path){
        auto last=path.end();
        for(auto here=path.begin();here!=path.end();here++){
          for(auto it:known){
            for(auto targ:target){
              keyname kl;
              kl.keyword=it.first;
              kl.targ   =targ;
              if(last!=path.end())
                kl.parname=*last;
              else
                kl.parname=" ";
              kl.actname=*here;
              learnOne(kl);
            }
          }
          last=here;
        }
      }
      virtual void getKey(const keyname &dep,string & name){
        name="dectree_prob_";
        name+=treename;
        const static string sp="_";
        
        if(!dep.keyword.empty())name+=sp+dep.keyword;
        if(!dep.targ.empty())   name+=sp+dep.targ;
        if(!dep.parname.empty())name+=sp+dep.parname;
        if(!dep.actname.empty())name+=sp+dep.actname;
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
        p->depend.clear();
        if(!depend.empty())
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
        keyname nl;
        nl.keyword=keyword;
        nl.targ=targ;
        nl.parname=parname;
        nl.actname="";
        
        string k;
        getKey(nl,k);
        char tm[32];
        snprintf(tm,32,"%d",time(NULL));
        db->Put(
          leveldb::WriteOptions(),k,tm
        );
        nl.actname=actname;
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
      inline void getRememberFromDB(const char * ks,set<probinfo> & pis,int num){
        leveldb::ReadOptions options;
        //options.snapshot = db->GetSnapshot();
        leveldb::Iterator* it = db->NewIterator(options);
        int i=0;
        for(it->Seek(ks);(it->Valid() && prefix_match(ks,it->key().data()));it->Next()){
          if(i>num)break;
          probinfo p;
          p.key=it->key().data();
          istringstream iss(it->value().data());
          iss>>p.times;
          iss>>p.actname;
          if(p.actname.empty())continue;
          pis.insert(p);
          i++;
        }
        delete it;
      }
      virtual void getRemember(const char * ks,set<probinfo> & pis){
        getRememberFromDB(ks,pis,searchTimes);
      }
      virtual void search(const keyname & dep,map<string,int> & acts){
        string k;
        getKey(dep,k);
        
        set<probinfo> pis;
        getRemember(k.c_str(),pis);
        
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
            keyname kl;
            kl.keyword=it.first;
            kl.targ   =targ;
            kl.parname=actname;
            kl.actname="";
            search(kl,acts);
          }
        }
        //直接用二分排序
        for(auto it:acts){
          probinfo p;
          p.actname=it.first;
          p.times=it.second;
          actl.insert(p);
        }
        acts.clear();//清空map，不然数据重复了，浪费
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
      activity(const char * initpath){
        L=luaL_newstate();
        this->luaopen();
        luaL_dofile(L,initpath);
      }
      ~activity(){
        lua_close(L);
      }
      static inline int isptr(lua_State * L,int p){
        return (lua_islightuserdata(L,p));
      }
      static inline void * toptr(lua_State * L,int p){
        if(lua_islightuserdata(L,p)){
          return lua_touserdata(L,p);
        }else{
          return NULL;
        }
      }
      static inline void pushptr(lua_State * L,void * ptr){
        lua_pushlightuserdata(L,ptr);
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
        lua_createtable(L,pinfo->depend.size()+1,0);
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
        string p=lua_tostring(L,2);
        self->doActivity(p);
        return 0;
      }
      static int lua_addactivity(lua_State * L){
        GETSELF;
        if(!lua_isstring(L,2))return 0;
        if(!lua_isstring(L,3))return 0;
        self->addActivity(
          lua_tostring(L,2),
          lua_tostring(L,3)
        );
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
          {"addactivity",lua_addactivity},
          {"getknown",lua_getknown},
          {"getenv",  lua_getenv},
          {"callactivity",lua_callactivity},
          NULL,NULL
        };
        utils::luaopen(L);
        lua_newtable(L);
        luaL_setfuncs(L, reg, 0);
        lua_setglobal(L,"ST");
      }
      void callactivity(const char * name){
        auto lt=lua_newthread(L);
        
        if(name){
          luaL_loadfile(lt,name);
          lua_getglobal(lt,"main");
        }else{
          lua_getglobal(lt,"onMissActivity");
        }
        
        if(!lua_isfunction(lt,-1))return;
        
        pushptr(lt,this);
        
        if(lua_pcall(lt,1,1,0)!=0){
          
        }else
          if(lua_isinteger(lt,-1))
            if(lua_tointeger(lt,-1)==0)
              lastDepth=nowDepth;
        lua_pop(lt,1);
      }
      virtual void addActivity(string actname,string scriptpath){
        string k="lua_activity_";
        k+=actname;
        db->Put(
          leveldb::WriteOptions(),k,scriptpath
        );
      }
      virtual void doActivity(const string & actname){
        string k="lua_activity_";
        k+=actname;
        env["ACT_NAME"]=actname;
        string v;
        if(
          !db->Get(
            leveldb::ReadOptions(),
            k,
            &v
          ).ok()
        ){
          env["SCRIPT_NAME"]="";
          callactivity(NULL);
          return;
        }
        if(v.empty()){
          env["SCRIPT_NAME"]="";
          callactivity(NULL);
        }else{
          env["SCRIPT_NAME"]=v;
          callactivity(v.c_str());
        }
      }
      virtual void findEvent(const keyname & kl){
        auto lt=lua_newthread(L);
        
        lua_getglobal(lt,"onLearnNew");
        if(!lua_isfunction(lt,-1))return;
        
        lua_createtable(lt,0,4);
        
        lua_pushstring(lt,"keyword");
        lua_pushstring(lt,kl.keyword.c_str());
        lua_settable(lt,-3);
        
        lua_pushstring(lt,"actname");
        lua_pushstring(lt,kl.actname.c_str());
        lua_settable(lt,-3);
        
        lua_pushstring(lt,"parname");
        lua_pushstring(lt,kl.parname.c_str());
        lua_settable(lt,-3);
        
        lua_pushstring(lt,"target");
        lua_pushstring(lt,kl.targ.c_str());
        lua_settable(lt,-3);
        
        lua_pcall(lt,1,0,0);
      }
      #undef GETSELF
    };
    class forget{
      public:
      char * corebuffer;  //核心缓冲区
                          //查找数据优先在这里查，找不到才到数据库
    };
    class dectree:public activity,public forget{
      
    };
  }
}
#endif