#ifndef atulocher_math
#define atulocher_math
namespace atulocher{
  namespace math{
    double invsqrt(double x){
      double xhalf=0.5f*x;
      int i=*(int*)&x;
      i=0x5f3759df-(i>>1);
      double y=*(double*)&i;
      return (y*(1.5f-xhalf*x*x));
    }
    double sqrt(double x){
      return (1.0f/invsqrt(x));
    }
  }
}
#endif