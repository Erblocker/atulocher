#ifndef atulocher_NN_MCPU
#define atulocher_NN_MCPU
#include "NN_layer.hpp"
#include <mutex>
#include <condition_variable>
#include <atomic>
#include "threadpool.hpp"
#include "mempool.hpp"
namespace atulocher{
namespace NN{

struct async_info{
  async_info * next;
  int i,j;
  void * tt;
};
mempool<async_info> async_pool;//内存池
class Layer_MCPU:public Layer {
  public:
  Layer_MCPU()=delete;
  void operator=(const Layer_MCPU&)=delete;
  Layer_MCPU(double eta, double momentum, int layer[],int szLayer, ActionType actionType):
    Layer(eta,momentum,layer,szLayer,actionType){
  
  }
  ~Layer_MCPU(){
    this->destroy();
  }
  struct ct_t{
    std::mutex mtx;
    std::condition_variable cv;
    std::atomic<int> num;
    Layer * L;
  };
  static void * Forward_cb(void *arg){
    auto asi=(async_info*)arg;
    auto tt=(ct_t*)asi->tt;
    
    std::unique_lock<std::mutex> lck(tt->mtx);
    
    auto pL=tt->L;
    int i=asi->i;
    int j=asi->j;
    pL->output[i + 1][j] = pL->act(pL->output[i + 1][j] + pL->theta[i][j]);
    tt->num--;
    
    tt->cv.notify_all();
    
    async_pool.del(asi);
  }
  virtual void Forward_node(int i,int j,ct_t * tt){
    auto asi=async_pool.get();
    asi->i=i;
    asi->j=j;
    asi->tt=tt;
    threadpool::add(Forward_cb,asi);
      
    //Forward_cb(asi);
  }
  virtual void Forward(){
    int lastIndex    = szLayer - 1;
    
    for (int i = 0; i < lastIndex; ++i){
        MatXMat(output[i], weights[i], output[i + 1], 1, layer[i + 1], layer[i]);
        
        ct_t tt;
        std::unique_lock<std::mutex> lck(tt.mtx);
        
        tt.num=layer[i + 1];
        tt.L=this;
        for (int j = 0; j < layer[i + 1]; ++j){
          Forward_node(i,j,&tt);
        }
        while(tt.num!=0){
          tt.cv.wait(lck);
        }
    }
    
  }
  
};
}
}
#endif