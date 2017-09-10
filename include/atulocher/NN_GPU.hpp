#ifndef atulocher_NN_GPU
#define atulocher_NN_GPU
#include "GPU.hpp"
#include "NN_layer.hpp"
namespace atulocher{
  namespace NN{
    class Layer_GPU:public Layer,private GPU{
      public:
      Layer_GPU(double eta, double momentum, int layer[],int szLayer, ActionType actionType):
        Layer(eta,momentum,layer,szLayer,actionType){
        this->initGPU();
      }
      Layer_GPU(Layer_GPU&)=delete;
      ~Layer_GPU(){
        this->destroy();
      }
      virtual void destroy(){
        this->freebuffer();
        this->cleanup();
      }
    };
  }
}
#endif