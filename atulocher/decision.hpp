#ifndef atulocher_decision
#define atulocher_decision
#include "searchbuffer.hpp"
#include "rpc.hpp"
#include "actscript.hpp"
namespace atulocher{
    class decision:public searchbuffer,public actscript{
        public:
            virtual void markActive(
                const char  *   data,
                const char  *   target,
                int             lastactivity,
                int             activity
            )=0;
            static void searchRoad(
                std::list<std::string> & res,
                const std::map<std::string,element*>    & dts,
                const std::map<std::string,bool>        & tgs,
                element * elm
            ){
                if(elm==NULL)
                    return;
                res.push_front(elm->val);
                for(auto it:elm->depend){
                    searchRoad(res,dts,tgs,it);
                }
            }
            virtual void getResult(std::list<std::string> & res){
                auto dts=getDatas();
                auto tgs=getTargets();
                for(auto itt:tgs){
                    if(itt.second){
                        auto itd=dts.find(itt.first);
                        if(itd==dts.end())
                            continue;
                        searchRoad(res,dts,tgs,itd->second);
                    }
                }
            }
            virtual void train(const std::list<int> & activity){
                int last=-1;
                auto dts=getDatas();
                auto tgs=getTargets();
                for(auto it:activity){
                    for(auto itd:dts){
                        for(auto itt:tgs){
                            markActive(
                                itd.second->info.name->c_str(),
                                itt.first.c_str(),
                                last,it
                            );
                        }
                    }
                    this->callactivity(it);
                    last=it;
                }
            }
            virtual void train(){
                auto dts=getDatas();
                auto tgs=getTargets();
                for(auto itd:dts){
                    if(!(itd.second->depend.empty())){
                        for(auto itdep:itd.second->depend){
                            for(auto itd2:dts){
                                for(auto itt:tgs){
                                    markActive(
                                        itd2.second->info.name->c_str(),
                                        itt.first.c_str(),
                                        itdep->activity,
                                        itd.second->activity
                                    );
                                }
                            }
                        }
                    }else{
                        for(auto itd2:dts){
                            for(auto itt:tgs){
                                markActive(
                                    itd2.second->info.name->c_str(),
                                    itt.first.c_str(),
                                    -1,
                                    itd.second->activity
                                );
                            }
                        }
                    }
                }
            }
            virtual SearchStatus callactivity(int activityname){
                std::string name;
                this->runScript(this->getActivity(activityname,name));
            }
        public:
            //overwrite functions of class actscript
            virtual void        setTargetMode   (const std::string & str,bool m=false){
                setTarget(str,m);
            }
            virtual void        autoUpdateTarget(const std::string & str){
                updateTarget(str);
            }
            virtual void        removeFromDatas (const std::string & str){
                removeData(str);
            }
            virtual dataInfo *  insertIntoDatas (const std::string & str){
                auto p=createData(str);
                if(p)
                    return &(p->info);
                else
                    return NULL;
            }
            virtual dataInfo *  getFromDatas    (const std::string & str){
                auto p=getData(str);
                if(p)
                    return &(p->info);
                else
                    return NULL;
            }
            virtual void        freeDataInfo    (dataInfo *){
                return;
            }
            virtual void        getAllDatas     (void(*cb)(dataInfo *,void*),void *arg){
                auto dts=getDatas();
                for(auto it:dts){
                    cb(&(it.second->info),arg);
                }
            }
            virtual void        getAllTargets   (void(*cb)(const char*,bool,void*),void *arg){
                auto tgs=getTargets();
                for(auto it:tgs){
                    cb(it.first.c_str(),it.second,arg);
                }
            }
    };
}
#endif