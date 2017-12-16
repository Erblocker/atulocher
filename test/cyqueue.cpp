#include <stdio.h>
#include <atulocher/cyqueue.hpp>
#include <atulocher/threadpool.hpp>
using namespace atulocher;
cyqueue<double> cq(128);
int main(){
  cq.push(2.0);
  double buf;
  cq.pop(&buf);
  printf("%f\n",buf);
  threadpool::add([](void*)->void*{
    sleep(3);
    cq.push(3.0);
    cq.push(4.0);
    cq.push(5.0);
  },NULL);
  threadpool::add([](void*)->void*{
    double buf;
    cq.pop(&buf);//block
    printf("%f\n",buf);
  },NULL);
  threadpool::add([](void*)->void*{
    double buf;
    cq.pop(&buf);//block
    printf("%f\n",buf);
  },NULL);
  cq.pop(&buf);//block
  printf("%f\n",buf);
  
}