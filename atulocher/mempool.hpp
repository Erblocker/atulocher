#ifndef atulocher_mempool
#define atulocher_mempool
#include <mutex>
#include <atomic>
namespace atulocher{
    template<typename T>
    class mempool_block{
      T * freed;
      std::mutex locker;
      std::atomic<int> rnum;//引用计数器
      public:
      void pickup(){
        rnum++;
      }
      void giveup(){
        rnum--;
        if(rnum==0){
          delete this;
          return;
        }
      }
      mempool_block(){
        freed=NULL;
        rnum=1;
      }
      ~mempool_block(){
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
    template<typename T>
    class mempool{
      protected:
      mempool_block<T> * parpool;
      public:
      mempool(){
        parpool=new mempool_block<T>;
      }
      mempool(const mempool<T> & pp){
        parpool=pp.parpool;
        parpool->pickup();
      }
      ~mempool(){
        parpool->giveup();
      }
      T * get(){
        return parpool->get();
      }
      void del(T * f){
        parpool->del(f);
      }
    };
    template<typename T>
    class mempool_auto{
      T * used;
      std::mutex locker;
      mempool<T> * par;
      int mnum;
      public:
      void bind(mempool<T> * p){
        par=p;
      }
      T * onew(){
        if(par)
          return par->get();
        else
          return new T;
      }
      void odel(T * ob){
        if(par)
          par->del(ob);
        else
          delete ob;
      }
      mempool_auto(){
        used=NULL;
        par =NULL;
        mnum=0;
      }
      ~mempool_auto(){
        this->clear();
      }
      void clear(){
        T * it1;
        T * it=used;
        int n=0;
        while(it){
          it1=it;
          it=it->gc_next;
          odel(it1);
          n++;
        }
        //printf("use:%d\tdel:%d\n",mnum,n);
      }
      T * get(){
        locker.lock();
          T * r;
          r=onew();
          r->gc_next=used;
          used =r;
          mnum++;
        locker.unlock();
        return r;
      }
    };
}
#endif