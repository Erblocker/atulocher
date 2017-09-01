#ifndef atulocher_mempool
#define atulocher_mempool
#include <mutex>
namespace atulocher{
    template<typename T>
    class mempool{
      T * freed;
      std::mutex locker;
      public:
      mempool(){
        freed=NULL;
      }
      ~mempool(){
        T * it1;
        T * it=freed;
        while(it){
          it1=it;
          it=it->next;
          delete it1;
        }
      }
      T * get(){
        locker.lock();
        if(freed){
          T * r=freed;
          freed=freed->next;
          locker.unlock();
          return r;
        }else{
          locker.unlock();
          return new T;
        }
      }
      void del(T * f){
        locker.lock();
        f->next=freed;
        freed=f;
        locker.unlock();
      }
    };
}
#endif