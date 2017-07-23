#ifndef atulocher_rand
#define atulocher_rand
#include <iostream>
#include <ctime>
namespace atulocher{
  class Rand{
    public:
    Rand(){
      srand(time(0));
    }
    double flo(){
      double r=rand()+rand()*0.00000001d;
      if(rand()<(RAND_MAX/2)){
        r=0.0d-r;
      }
      return r;
    }
  }rand;
}
#endif