#ifndef yrssf_atulocher_actscript
#define yrssf_atulocher_actscript
#include "luapool.hpp"
#include "rpc.hpp"
#include <string>
#include <vector>
#include <list>
namespace atulocher{
  class actscript{
    public:
    void getActivity(double * arr,int len,std::string & name){
      RakNet::BitStream res,ret;
      
      res<<len;
      for(int i=0;i<len;i++)res<<arr[i];
      
      rpc.call("PSPredict",&res,&ret);
      
      
      RakNet::RakString data;
      int offset= ret.GetReadOffset();
      bool read = ret.ReadCompressed(data);
      
      name=data.C_String();
    }
  };
}
#endif
