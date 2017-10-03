#ifndef atulocher_NN_RNN
#define atulocher_NN_RNN
#include "NN_layer.hpp"
namespace atulocher{
  namespace NN{
    class Layer_rnn:public Layer{
      public:
      int * from;
      Layer_rnn(double eta, double momentum, int layer[],int szLayer, ActionType actionType,const int *f):Layer(){
        from=new int[szLayer];
        for(int i=0;i<szLayer;i++){
          from[i]=f[i];
        }
        this->construct(eta,momentum,layer,szLayer,actionType);
      }
      Layer_rnn(double eta, double momentum, int layer[],int szLayer, ActionType actionType):Layer(){
        from=new int[szLayer];
        for(int i=0;i<szLayer;i++){
          from[i]=-1;
        }
        from[0]=szLayer-1;
        this->construct(eta,momentum,layer,szLayer,actionType);
      }
      virtual void construct(double eta, double momentum, int layer[],int szLayer, ActionType actionType){
        int * layer_t=new int[szLayer];
        for(int i=0;i<szLayer;i++){
          int fr=from[i];
          if(fr>=0)
            layer_t[i]=layer[i]+layer[fr];
          else
            layer_t[i]=layer[i];
        }
        this->CreateLayer(eta,momentum,layer_t,szLayer,actionType);
        delete [] layer_t;
      }
      void freefrom(){
        delete [] from;
      }
      virtual void destroy(){
        this->freefrom();
        this->freebuffer();
      }
      virtual int getInputSize(){
        int fr=from[0];
        if(fr>=0)
          return layer[0]-layer[fr];
        else
          return layer[0];
      }
      virtual int getOutputSize(){
        int fr=from[szLayer-1];
        if(fr>=0)
          return layer[szLayer-1]-layer[fr];
        else
          return layer[szLayer-1];
      }
      virtual int getLSize(int i){
        int fr=from[i];
        if(fr>=0)
          return layer[i]-layer[fr];
        else
          return layer[i];
      }
      virtual void LoadInput(double input[]){
        int len=getInputSize();
        for (int i = 0; i < len; ++i)
          output[0][i] = input[i];
      }
      virtual void LoadTarget(double target[]){
        int lastIndex  = szLayer - 1;
        double *delta_p  = delta[lastIndex - 1];
        double *output_p = output[lastIndex];
        int len=getOutputSize();
        for (int i = 0; i < len; ++i)
          delta_p[i] = actdiff(output_p[i])*(target[i] - output_p[i]);
        
      }
      virtual void reset(){
        for (int i = 0; i < szLayer; ++i){
          int fr=from[i];
          if(fr<0)continue;
          int len=layer[i];
          for (int j = getLSize(i)+1; j < len; ++j){
            output[i][j]=0.0d;
          }
        }
      }
      virtual void outbackup(){
        for (int i = 0; i < szLayer; ++i){
          int fr=from[i];
          if(fr<0)continue;
          int len=layer[i];
          int k=0;
          for (int j = getLSize(i)-1; j < len; ++j){
            output[i][j]=output[fr][k];
            k++;
          }
        }
      }
      virtual void Forward(){
        int lastIndex    = szLayer - 1;
        for (int i = 0; i < lastIndex; ++i){
          int len=getLSize(i + 1);
          MatXMat(output[i], weights[i], output[i + 1], 1, len, layer[i]);
          for (int j = 0; j < len; ++j){
            output[i + 1][j] = act(output[i + 1][j] + theta[i][j]);
          }
        }
      }
      virtual void AdjustWeights(){
        int lastIndex = this->szLayer - 1;
        int len;
        for (int i = lastIndex-1; i > 0; --i){
            len=getLSize(i + 1);
            MatXMat(weights[i], delta[i], delta[i - 1], layer[i], 1, len );
            for (int j = 0; j < layer[i]; ++j){
              delta[i - 1][j] *= actdiff(output[i][j]);
              delta[i - 1][j] +=makeDelta*lfrand()*(brand()?1:-1);
            }
        }
    
        for (int i = 0; i < lastIndex; ++i){
          len=getLSize(i + 1);
          for (int j = 0; j < layer[i]; ++j){
            for (int k = 0; k < len; ++k){
                int pos = j*layer[i + 1] + k;
                preWeights[i][pos] = momentum * preWeights[i][pos] + eta * delta[i][k] * output[i][j];
                weights[i][pos] += preWeights[i][pos];
            }
          }
          for (int j = 0; j < len; ++j){
            preTheta[i][j] = momentum*preTheta[i][j] + eta*delta[i][j];
            theta[i][j] += preTheta[i][j];
          }
        }
      }
      virtual void train(double input[], double target[],bool re=false){
        if(act && actdiff){
          if(re)
            reset();
          else
            outbackup();
          LoadInput(input);
          Forward();
          LoadTarget(target);
          AdjustWeights();
        }
      }
      virtual void predict(double input[],double output[],bool re=false){
        if(act && actdiff){
          if(re)
            reset();
          else
            outbackup();
          int lastIndex = szLayer - 1;
          LoadInput(input);
          Forward();
          double * res = this->output[lastIndex];
        int len=getOutputSize();
        for (int i = 0; i < len; ++i)
            output[i] = res[i];
        }
      }
    };
    class Layer_lstm{
      public:
      Layer_lstm(){
        printf("不会\n");
      }
    };
  }
}
#endif