#ifndef atulocher_active
#define atulocher_active
#include "NN.hpp"
namespace atulocher{
  namespace active{
    class active{
      NN::RNN * net;
      public:
      void(*onGetEvent)(double*);
      active(){}
    };
  }
}
#endif