#ifndef yrssf_atulocher_actscript
#define yrssf_atulocher_actscript
#include "object.hpp"
#include "luapool.hpp"
#include "rpc.hpp"
#include <string>
#include <vector>
#include <list>
namespace atulocher {
class actscript {
    public:
    struct dataInfo{
        const std::string     * name;
        std::string           * val;
        atuobj::object        * obj;
        virtual void getDepend(void(*cb)(const dataInfo*,void*),void *arg)=0;
    };
    std::string & getActivity(double * arr,int len,std::string & name) {
        RakNet::BitStream res,ret;

        res<<len;
        for(int i=0; i<len; i++)res<<arr[i];

        rpc.call("Act_GetByVec",&res,&ret);


        RakNet::RakString data;
        int offset= ret.GetReadOffset();
        bool read = ret.ReadCompressed(data);

        name=data.C_String();
        
        return name;
    }
    std::string & getActivity(int activityname,std::string & name) {
        RakNet::BitStream res,ret;
        res<<activityname;

        rpc.call("Act_GetById",&res,&ret);

        RakNet::RakString data;
        int offset= ret.GetReadOffset();
        bool read = ret.ReadCompressed(data);

        name=data.C_String();
        
        return name;
    }
    void runScript(const std::string & name){
        
    }
    public:
    virtual void        setTargetMode   (const std::string & str,bool m=false)       =0;
    virtual void        autoUpdateTarget(const std::string & str)                    =0;
    virtual void        removeFromDatas (const std::string & str)                    =0;
    virtual dataInfo *  insertIntoDatas (const std::string & str)                    =0;
    virtual dataInfo *  getFromDatas    (const std::string & str)                    =0;
    virtual void        freeDataInfo    (dataInfo *)                                 =0;
    virtual void        getAllDatas     (void(*cb)(dataInfo * ,void*),void *arg)     =0;
    virtual void        getAllTargets   (void(*cb)(const char*,bool,void*),void *arg)=0;
};
}
#endif
