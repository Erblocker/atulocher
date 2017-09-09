#ifndef atulocher_NN_layer
#define atulocher_NN_layer
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <memory.h>
#include <unistd.h>
#include <sstream>
#include <iostream>
#include <string>
namespace atulocher{
namespace NN{
    
typedef enum {
  CUSTOM=0,
  SIGMOD=1
} ActionType;

typedef double(*Function)(double);

inline void MatXMat(double mat1[], double mat2[], double output[], int row, int column, int lcolrrow){
    for (int i = 0; i < row; ++i)
        for (int j = 0; j < column; ++j){
            int pos = column * i + j;
            output[pos] = 0;
            for (int k = 0; k < lcolrrow; ++k)
                output[pos] += mat1[lcolrrow * i + k] * mat2[column * k + j];
        }
}

//随机生成－1.0～1.0之间的随机浮点数
inline double lfrand(){
     
    static int randbit = 0;
    if (!randbit){
        srand((unsigned)time(0));
        for (int i = RAND_MAX; i; i >>= 1, ++randbit);
    }
    unsigned long long lvalue = 0x4000000000000000L;
    int i = 52 - randbit;
    for (; i > 0; i -= randbit)
        lvalue |= (unsigned long long)rand() << i;
    lvalue |= (unsigned long long)rand() >> -i;
    return *(double *)&lvalue - 3;
}
 
double Sigmod(double x){
    return 1 / (1 + exp(-x));
}
 
double SigmodDiff(double y){
    return y*(1 - y);
}


class Layer{
  static int GetCbSize(int szLayer,int layer[]){
      int cbSize = 0;
      cbSize += sizeof(int)*szLayer;
      cbSize += sizeof(double *)*(szLayer*6-5);
      cbSize += sizeof(double)*layer[0];
      for (int i = 1; i < szLayer; ++i){
          cbSize += sizeof(double)*layer[i] * layer[i - 1]*2;
          cbSize += sizeof(double)*layer[i]*6;
      }
      return cbSize;
  }
  public:
    int szLayer;            //层数
    double eta;
    double momentum;
    int *layer;             //每层的结点数
    ActionType actionType;  //激活函数类型
    Function act;           //激活函数
    Function actdiff;       //激活函数的导数
    double **weights;       //权值
    double **preWeights;    //前一时刻的权值
    double **delta;         //误差值
    double **theta;         //阈值
    double **preTheta;      //前一时刻的阈值
    double **output;        //每层结点的输出值
    private:
    void *   buffer;        //用于存储结点数、权值、前一时刻的权值、误差值、阈值、前一时刻的阈值、结点输出值的空间
  public:
  virtual void InitLayer(){
    Layer *pLayer=this;
    switch (pLayer->actionType) {
        case SIGMOD:
            pLayer->act=Sigmod;
            pLayer->actdiff = SigmodDiff;
            break;
             
        default:
            pLayer->act = 0;
            pLayer->actdiff = 0;
            break;
    }
     
    int szLayer = pLayer->szLayer;
    pLayer->layer = (int *)pLayer->buffer;
    pLayer->output = (double **)(pLayer->layer+szLayer);
    pLayer->delta = pLayer->output+szLayer;
    pLayer->weights = pLayer->delta + szLayer - 1;
    pLayer->theta = pLayer->weights + szLayer - 1;
    pLayer->preWeights = pLayer->theta + szLayer - 1;
    pLayer->preTheta = pLayer->preWeights + szLayer - 1;
     
    *(pLayer->output) = (double *)(pLayer->preTheta + szLayer - 1);
    int *layer = pLayer->layer;
    for (int i = 0; i < szLayer; ++i)
        pLayer->output[i+1]=pLayer->output[i]+layer[i];
    for (int i=0;i<szLayer - 1;++i)
        pLayer->delta[i+1]=pLayer->delta[i]+layer[i+1];
    for (int i = 0; i < szLayer - 1; ++i)
        pLayer->weights[i+1]=pLayer->weights[i]+layer[i] * layer[i + 1];
    for(int i=0;i<szLayer - 1;++i)
        pLayer->theta[i+1]=pLayer->theta[i]+layer[i+1];
    long long tmp = pLayer->theta[szLayer - 1]-pLayer->weights[0];
    for(int i=0;i<szLayer - 1;++i){
        pLayer->preWeights[i]=pLayer->weights[i]+tmp;
        pLayer->preTheta[i] = pLayer->theta[i]+tmp;
    }
  }
  virtual void CreateLayer(double eta, double momentum, int layer[],int szLayer, ActionType actionType){
    Layer *pLayer=this;
    int cbSize = GetCbSize(szLayer, layer);
    buffer = malloc(cbSize);
    bzero(buffer,cbSize);
    pLayer->eta = eta;
    pLayer->momentum = momentum;
    pLayer->szLayer = szLayer;
    pLayer->actionType = actionType;
    pLayer->layer = (int *)pLayer->buffer;
    for(int i=0;i<szLayer;++i)
        pLayer->layer[i] = layer[i];
    InitLayer();
    for(double *i=pLayer->weights[0];i!=pLayer->preWeights[0];++i)
        *i=lfrand();
    //for(double *i=pLayer->preWeights[0];i!=(double *)((unsigned char *)pLayer+cbSize);++i)
    //    *i=0;
    //加载大神经网络时会Segmentation fault，已经由前面的bzero取代
  }
  void freebuffer(){
    free(buffer);
  }
  virtual void destroy(){
    freebuffer();
  }
  Layer()=delete;
  void operator=(const Layer&)=delete;
  Layer(double eta, double momentum, int layer[],int szLayer, ActionType actionType){
    this->CreateLayer(eta,momentum,layer,szLayer,actionType);
  }
  ~Layer(){
    this->destroy();
  }
  virtual void LoadInput(double input[]){
    for (int i = 0; i < layer[0]; ++i)
      output[0][i] = input[i];
  }
  virtual void LoadTarget(double target[]){
    int lastIndex  = szLayer - 1;
    double *delta_p  = delta[lastIndex - 1];
    double *output_p = output[lastIndex];
    
    for (int i = 0; i < layer[lastIndex]; ++i)
      delta_p[i] = actdiff(output_p[i])*(target[i] - output_p[i]);
    
  }
  virtual void Forward(){
    int lastIndex    = szLayer - 1;
    
    for (int i = 0; i < lastIndex; ++i){
        MatXMat(output[i], weights[i], output[i + 1], 1, layer[i + 1], layer[i]);
        for (int j = 0; j < layer[i + 1]; ++j){
            output[i + 1][j] = act(output[i + 1][j] + theta[i][j]);
        }
    }
    
  }
  virtual void AdjustWeights(){
    int lastIndex = this->szLayer - 1;
    
    for (int i = lastIndex-1; i > 0; --i){
        MatXMat(weights[i], delta[i], delta[i - 1], layer[i], 1, layer[i + 1]);
        for (int j = 0; j < layer[i]; ++j)
            delta[i - 1][j] *= actdiff(output[i][j]);
    }
    
    for (int i = 0; i < lastIndex; ++i){
        for (int j = 0; j < layer[i]; ++j){
            for (int k = 0; k < layer[i + 1]; ++k){
                int pos = j*layer[i + 1] + k;
                preWeights[i][pos] = momentum * preWeights[i][pos] + eta * delta[i][k] * output[i][j];
                weights[i][pos] += preWeights[i][pos];
            }
        }
        for (int j = 0; j < layer[i+1]; ++j){
            preTheta[i][j] = momentum*preTheta[i][j] + eta*delta[i][j];
            theta[i][j] += preTheta[i][j];
        }
    }
  }
  virtual void train(double input[], double target[]){
    if(act && actdiff){
        LoadInput(input);
        Forward();
        LoadTarget(target);
        AdjustWeights();
    }
  }
  virtual void predict(double input[],double output[]){
    if(act && actdiff){
        int lastIndex = szLayer - 1;
        LoadInput(input);
        Forward();
        double * res = this->output[lastIndex];
        
        for (int i = 0; i < layer[lastIndex]; ++i)
            output[i] = res[i];
    }
  }
  virtual void toString(std::string & out){
    char buf[100];
    out="";
    for(int i=0;i<this->szLayer;i++){
      for(int j=0;j<this->layer[i];j++){
        snprintf(buf,100,"%lf %lf ",
          this->weights[i][j],
          this->theta[i][j]
        );
        out+=buf;
      }
    }
  }
  virtual void loadString(const std::string & in){
    std::istringstream iss(in);
    for(int i=0;i<this->szLayer;i++){
      for(int j=0;j<this->layer[i];j++){
        iss>>this->weights[i][j];
        iss>>this->theta[i][j];
      }
    }
  }
};
///////////////////////
}
}
#endif