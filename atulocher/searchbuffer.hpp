#ifndef atulocher_searchbuffer
#define atulocher_searchbuffer
#include "mempool.hpp"
#include "bayes.hpp"
#include "object.hpp"
#include <set>
#include <map>
#include <list>
#include <stdio.h>
#include <memory.h>
#include <stdlib.h>
#include <string>
#include <algorithm>
#include "actscript.hpp"
namespace atulocher{
    class searchbuffer{
        public:
            typedef enum{
                CONTINUE,
                FINISH,
                STOP,
                NEXT,
                OK
            }SearchStatus;
            class element{
                friend class mempool<element>;
                friend class searchbuffer;
                protected:
                    element     *       next;
                    std::string         name;
                public:
                    std::string         val;
                    std::set<element*>  depend;
                    int                 activity;
                    atuobj::object      obj;
                    class Info:public actscript::dataInfo{
                        public:
                        element * self;
                        virtual void getDepend(void(*cb)(const dataInfo*,void*),void *arg){
                            for(auto it:self->depend){
                                if(it)
                                    cb(&(it->info),arg);
                            }
                        }
                    }info;
                protected:
                    void construct(){
                        name.clear();
                        val.clear();
                        obj.clear();
                        depend.clear();
                        activity=-1;
                        
                        info=Info();
                        info.name=&name;
                        info.val =&val;
                        info.obj =&obj;
                        info.self=this;
                    }
                    void destruct(){
                        name.clear();
                        val.clear();
                        obj.clear();
                        depend.clear();
                    }
            };
            virtual void search(){
                this->lastdeep=0;
                searchStep();
            }
            virtual bool searchStep(int lastactivity=-1,int deep=0){
                if(deep==0)
                    this->lastdeep=0;
                else{
                    if(deep-(this->lastdeep)>=this->maxemptydeep || deep>=maxdeep){
                        return false;
                    }
                }
                std::map<int,int> actives;
                bool stop=false,finish=false;
                int ibuf1,ibuf2,S;
                this->times--;
                for(auto it1:targets){
                    if(it1.second)      continue;
                    if(stop)            break;  //searching has been stoped
                    if(this->times<0)   break;  //times has been run out
                    
                    S=0;
                    actives.clear();    //reset
                    for(auto it2:datas){
                        struct tmp_stu{
                            std::map<int,int> * actives;
                            int * S;
                        }tmp;
                        tmp.S=&S;
                        tmp.actives=&actives;
                        getActiveByPair(
                            it2.first.c_str(),
                            it1.first.c_str(),
                            lastactivity,
                            [](int active,int times,void * arg)->void{
                                auto self=(tmp_stu*)arg;
                                (*(self->actives))  [active]+=times;
                                (*(self->S))                +=times;
                            },&tmp
                        );
                    }
                    if(S==0)continue;
                    
                    auto actlist=new std::pair<int,int> [actives.size()];
                    auto actptrs=new std::pair<int,int>*[actives.size()];
                    int activelen=actives.size();
                    int i=0;
                    for(auto it:actives){
                        actlist[i].first=it.first;
                        actlist[i].second=it.second;
                        actptrs[i]=&actlist[i];
                        ++i;
                    }
                    std::sort(actptrs,actptrs+activelen,[](std::pair<int,int>* p1,std::pair<int,int>* p2){
                        return p1->second > p2->second;
                    });
                    actives.clear();    //data have been wrote into actlist
                    
                    for(i=0;i<activelen;i++){
                        if(((double)(actptrs[i]->second)/(double)S)<maxP)
                            break;
                        if(it1.second)
                            break;
                        SearchStatus status=callactivity(actptrs[i]->first);
                        if(status==CONTINUE){
                            if((deep-lastdeep)<maxemptydeep && deep<maxdeep){
                                
                                if(
                                    searchStep(
                                        actptrs[i]->first,      //this activity
                                        deep+1                  //this deep
                                    )
                                ){
                                    status=checkStatus(actptrs[i]->first);
                                    if(status==STOP){
                                        stop=true;
                                        break;
                                    }else
                                    if(status==FINISH){
                                        stop=true;
                                        finish=true;
                                        break;
                                    }else
                                    if(status==NEXT){
                                        break;
                                    }
                                }
                            }
                        }else
                        if(status==FINISH){
                            stop=true;
                            finish=true;
                            break;
                        }else
                        if(status==STOP){
                            stop=true;
                            break;
                        }else
                        if(status==OK){
                            this->lastdeep=deep;
                            break;
                        }
                    }
                    
                    delete [] actlist;
                    delete [] actptrs;
                    return finish;
                }
            }
            virtual SearchStatus checkStatus(int active){
                if(waitforsolve==0)return FINISH;
                return CONTINUE;
            }
        public:
            virtual void getActiveByPair(
                const char  *   data,
                const char  *   target,
                int             lastactivity,
                void(       *   callback)(int,int,void*),
                void        *   arg
            )=0;
            virtual SearchStatus callactivity(int activityname)=0;
            int maxemptydeep,maxdeep;
            double maxP;
            int times;      //操作次数（行动力）
        protected:
            mempool<element> pool;
            std::map<std::string,element*>  datas;
            std::map<std::string,bool>      targets;    //true:the target have been solved
            int waitforsolve;
            int lastdeep;
        public:
            inline const std::map<std::string,element*> & getDatas(){
                return datas;
            }
            inline const std::map<std::string,bool> & getTargets(){
                return targets;
            }
            inline void updateTarget(const std::string & str){
                if(datas.find(str)==datas.end())
                    setTarget(str,false);
                else
                    setTarget(str,true);
            }
            inline void setTarget(const std::string & str,bool m=false){
                auto it=this->targets.find(str);
                if(it==this->targets.end()){
                    if(m==false)
                        this->waitforsolve++;
                    targets[str]=m;
                }else{
                    if(it->second){                 //if the target have been solved
                        if(!m){                     //if you set the target as "no solve"
                            this->waitforsolve++;
                        }
                    }else{
                        if(m){                      //if you set the target as "solved"
                            this->waitforsolve--;
                        }
                    }
                    it->second=m;
                }
            }
            inline element * getData(const std::string & str){
                auto it=this->datas.find(str);
                if(it==this->datas.end())
                    return NULL;
                else
                    return it->second;
            }
            inline void removeData(const std::string & str){
                auto it=this->datas.find(str);
                if(it!=this->datas.end()){
                    it->second->destruct();
                    pool.del(it->second);
                    this->datas.erase(it);
                }
            }
            inline element * createData(const std::string & str){
                auto it=this->datas.find(str);
                if(it!=this->datas.end())   //data have existed
                    return NULL;
                else{
                    auto em=this->pool.get();
                    em->construct();
                    em->name=str;
                    this->datas[str]=em;
                    return em;
                }
            }
        public:
            searchbuffer(){
                
            }
            ~searchbuffer(){
                for(auto it:datas){
                    if(it.second){
                        pool.del(it.second);
                    }
                }
            }
            
    };
}
#endif
