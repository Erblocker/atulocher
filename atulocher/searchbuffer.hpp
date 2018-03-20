#ifndef atulocher_searchbuffer
#define atulocher_searchbuffer
#include "mempool.hpp"
#include <set>
#include <map>
#include <list>
#include <stdio.h>
#include <memory.h>
#include <stdlib.h>
#include <string>
#include <algorithm>
namespace atulocher{
    class searchbuffer{
        public:
            typedef enum{
                CONTINUE,
                FINISH,
                STOP,
                NEXT
            }SearchStatus;
            struct element{
                element     *       next;
                char                name[128];
                std::string         val;
                std::set<element*>  depend;
                void construct(){
                    bzero(name,128);
                    val.clear();
                    depend.clear();
                }
                void destruct(){
                    val.clear();
                    depend.clear();
                }
            };
            virtual bool searchStep(int last=-1,int deep=0){
                if(deep==0)lastdeep=0;
                std::map<int,int> actives;
                bool stop=false,finish=false;
                int ibuf1,ibuf2,S;
                this->times--;
                for(auto it1:targets){
                    if(stop)            break;  //searching has been stoped
                    if(this->times<0)   break;  //times has been run out
                    
                    S=0;
                    actives.clear();    //reset
                    
                    for(auto it2:datas){
                        getActiveByPair(it2.first.c_str(),it1.c_str(),last,&ibuf1,&ibuf2);
                        actives[ibuf1]+=ibuf2;
                        S+=ibuf2;
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
                        if(((double)(actptrs[i]->second)/(double)S)<maxP)break;
                        SearchStatus status=callactivity(actptrs[i]->first);
                        if(status==CONTINUE){
                            if((deep-lastdeep)<maxemptydepth && deep<maxdeep){
                                if(searchStep(actptrs[i]->first,deep+1)){
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
                        }
                    }
                    
                    delete [] actlist;
                    delete [] actptrs;
                    return finish;
                }
            }
            virtual SearchStatus checkStatus(int active){
                
            }
        public:
            virtual void getActiveByPair(const char * d,const char * t,int last,int * active,int * times)=0;
            virtual SearchStatus callactivity(int)=0;
            int maxemptydepth,maxdeep;
            double maxP;
            int times;      //操作次数（行动力）
        protected:
            mempool<element> pool;
            std::map<std::string,element*>  datas;
            std::set<std::string>           targets;
            int lastdeep;
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
