#include <stdio.h>
#include <atulocher/cyqueue.hpp>
using namespace atulocher;
int main(){
  cyqueue<double> cq(128);
  cq.push(2.0);
  double buf;
  cq.pop(&buf);
  printf("%f\n",buf);
}