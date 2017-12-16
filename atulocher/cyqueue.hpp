#ifndef atulocher_cyqueue
#define atulocher_cyqueue
#include <atomic>
#include <mutex>
#include <condition_variable>
namespace atulocher{
  template<typename T>
  class cyqueue{
    private:
    std::condition_variable wcv,rcv;
    std::mutex mtx,rl,wl;
    T * arr;
    std::atomic<int> rp,wp;
    int length;
    int next(int pp){
      if(pp+1==length)
        return 0;
      else
        return pp+1;
    }
    bool push_unsafe(T v){
      if(next(wp)==rp)
        return false;
      arr[wp]=v;
      wp=next(wp);
      return true;
    }
    bool pop_unsafe(T * v){
      if(next(rp)==wp)
        return false;
      *v=arr[rp+1];
      rp=next(rp);
      return true;
    }
    public:
    void push(T v){
      std::unique_lock<std::mutex> lck(mtx);
      while(1){
        if(push_noblock(v))
          return;
        else{
          wcv.wait(lck);
        }
      }
    }
    void pop(T * v){
      std::unique_lock<std::mutex> lck(mtx);
      while(1){
        if(pop_noblock(v))
          return;
        else{
          rcv.wait(lck);
        }
      }
    }
    bool push_noblock(T v){
      wl.lock();
      auto r=push_unsafe(v);
      wl.unlock();
      rcv.notify_all();
      return r;
    }
    bool pop_noblock(T * v){
      rl.lock();
      auto r=pop_unsafe(v);
      rl.unlock();
      wcv.notify_all();
      return r;
    }
    cyqueue(int len){
      if(len<4)return;
      length=len;
      arr=new T[len];
      rp=0;
      wp=1;
    }
    ~cyqueue(){
      delete [] arr;
    }
  };
}
#endif