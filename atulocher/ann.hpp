#ifndef atulocher_ann
#define atulocher_ann
#include "rpc.hpp"
#include <sstream>
namespace atulocher{
  namespace ann{
    typedef int FD;
    int Predict(FD network,
                double * input_data,
                double * output_data,
                int data_size,
                int tm,int maxlen){
      RakNet::BitStream res,ret;
      double buf;
      buf=data_size;     res<<buf;
      buf=fd;            res<<buf;
      buf=tm;            res<<buf;
      buf=maxlen;        res<<buf;
      int i;
      for(i=0;i<data_size;i++) res<<input_data[i];
      rpc.call("PSPredict",res,ret);
      for(i=0;i<data_size;i++) ret>>output_data[i];
    }
    
    
  }
}
#endif
