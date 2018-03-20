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
            std::list<std::string> searchstack;
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
            virtual void searchStep(std::list<int> & buffer){
                std::map<int,int> actives;
                int ibuf1,ibuf2;
                for(auto it1:targets){
                    for(auto it2:datas){
                        getActiveByPair(it2.first.c_str(),it1.c_str(),&ibuf1,&ibuf2);
                        actives[ibuf1]+=ibuf2;
                    }
                }
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
                for(i=0;i<activelen;i++){
                    buffer.push_back(actptrs[i]->first);
                }
                delete [] actlist;
                delete [] actptrs;
                
            }
        public:
            virtual void getActiveByPair(const char * d,const char * t,int * active,int * times)=0;
        private:
            mempool<element> pool;
            std::map<std::string,element*>  datas;
            std::set<std::string>           targets;
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
