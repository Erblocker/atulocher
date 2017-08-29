#ifndef atulocher_NN
#define atulocher_NN
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
 
void MatXMat(double mat1[], double mat2[], double output[], int row, int column, int lcolrrow){
    for (int i = 0; i < row; ++i)
        for (int j = 0; j < column; ++j){
            int pos = column * i + j;
            output[pos] = 0;
            for (int k = 0; k < lcolrrow; ++k)
                output[pos] += mat1[lcolrrow * i + k] * mat2[column * k + j];
        }
}

//随机生成－1.0～1.0之间的随机浮点数
double lfrand(){
     
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

struct Layer{
    int cbSize;             //神经网络所占用的内存空间
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
    void* buffer[0];        //用于存储结点数、权值、前一时刻的权值、误差值、阈值、前一时刻的阈值、结点输出值的空间
    
  void LoadInput(double input[]){
    for (int i = 0; i < layer[0]; ++i)
      output[0][i] = input[i];
  }
  void LoadTarget(double target[]){
    int lastIndex  = szLayer - 1;
    double *delta_p  = delta[lastIndex - 1];
    double *output_p = output[lastIndex];
    
    for (int i = 0; i < layer[lastIndex]; ++i)
      delta_p[i] = actdiff(output_p[i])*(target[i] - output_p[i]);
    
  }
  void Forward(){
    int lastIndex    = szLayer - 1;
    
    for (int i = 0; i < lastIndex; ++i){
        MatXMat(output[i], weights[i], output[i + 1], 1, layer[i + 1], layer[i]);
        for (int j = 0; j < layer[i + 1]; ++j)
            output[i + 1][j] = act(output[i + 1][j] + theta[i][j]);
    }
    
  }
  void AdjustWeights(){
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
  void train(double input[], double target[]){
    if(act && actdiff){
        LoadInput(input);
        Forward();
        LoadTarget(target);
        AdjustWeights();
    }
  }
  void predict(double input[],double output[]){
    if(act && actdiff){
        int lastIndex = szLayer - 1;
        LoadInput(input);
        Forward();
        double * res = this->output[lastIndex];
        
        for (int i = 0; i < layer[lastIndex]; ++i)
            output[i] = res[i];
    }
  }
};
 
static int GetCbSize(int szLayer,int layer[]){
    int cbSize = sizeof(Layer);
    cbSize += sizeof(int)*szLayer;
    cbSize += sizeof(double *)*(szLayer*6-5);
    cbSize += sizeof(double)*layer[0];
    for (int i = 1; i < szLayer; ++i){
        cbSize += sizeof(double)*layer[i] * layer[i - 1]*2;
        cbSize += sizeof(double)*layer[i]*4;
    }
    return cbSize;
}
 
void InitLayer(void *buffer){
    Layer *pLayer = (Layer *)buffer;
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
 
int SaveLayer(Layer *pLayer,const char *filename){
    if(!pLayer) return 0;
    FILE *fp =0;
    if((fp = fopen(filename, "wb+"))){
        fprintf(fp,"%d %lf %lf %d\n",
          pLayer->szLayer,
          pLayer->eta,
          pLayer->momentum,
          (int)pLayer->actionType
        );
        
        for(int i=0;i<pLayer->szLayer;i++){
          fprintf(fp,"%d ",pLayer->layer[i]);
        }
        fprintf(fp,"\n");
        
        for(int i=0;i<pLayer->szLayer;i++){
          for(int j=0;j<pLayer->layer[i];j++){
            
            fprintf(fp,"%lf %lf\n",
              pLayer->weights[i][j],
              pLayer->theta[i][j]
            );
            
          }
        }
        
        return fclose(fp);
    }
    return 0;
}
Layer* CreateLayer(double eta, double momentum, int layer[],int szLayer, ActionType actionType);
Layer* LoadLayer(const char *filename){
    if(!filename) return 0;
    FILE *fp = 0;
    if((fp=fopen(filename, "r"))){
        char buf[4096];
        fgets(buf,4096,fp);
        std::istringstream iss(buf);
        int szLayer;
        double eta;
        double momentum;
        ActionType actionType;
        int atype;
        iss >> szLayer;
        iss >> eta;
        iss >> momentum;
        iss >> atype;
        actionType=(ActionType)atype;
        if(szLayer<=0)goto end;
        auto layer=new int[szLayer];
        fgets(buf,4096,fp);
        std::istringstream issb(buf);
        for(int i=0;i<szLayer;i++){
          int li;
          issb>>li;
          if(li<=0){
            delete [] layer;
            goto end;
          }
          layer[i]=li;
        }
       Layer* pLayer=CreateLayer(eta,momentum,layer,szLayer,SIGMOD);
        for(int i=0;i<pLayer->szLayer;i++){
          for(int j=0;j<pLayer->layer[i];j++){
            
            
            fgets(buf,4096,fp);
            std::istringstream issc(buf);
              issc>>pLayer->weights[i][j];
              issc>>pLayer->theta[i][j];
            
            
          }
        }
        delete [] layer;
        fclose(fp);
        return pLayer;
    }
    end:
    fclose(fp);
    return 0;
}
 
Layer* CreateLayer(double eta, double momentum, int layer[],int szLayer, ActionType actionType){
    int cbSize = GetCbSize(szLayer, layer);
    Layer* pLayer = (Layer *)malloc(cbSize);
    pLayer->cbSize = cbSize;
    pLayer->eta = eta;
    pLayer->momentum = momentum;
    pLayer->szLayer = szLayer;
    pLayer->actionType = actionType;
    pLayer->layer = (int *)pLayer->buffer;
    for(int i=0;i<szLayer;++i)
        pLayer->layer[i] = layer[i];
    InitLayer(pLayer);
    for(double *i=pLayer->weights[0];i!=pLayer->preWeights[0];++i)
        *i=lfrand();
    for(double *i=pLayer->preWeights[0];i!=(double *)((unsigned char *)pLayer+cbSize);++i)
        *i=0;
    return pLayer;
}
 
int DestroyLayer(Layer *pLayer){
    if (!pLayer) return 0;
    free(pLayer);
    return 1;
}
 
void ToBinary(unsigned x, unsigned n,double output[]){
    for (unsigned i = 0, j = x; i < n; ++i, j >>= 1)
        output[i] = j & 1;
}
 
unsigned FromBinary(double output[],unsigned n){
    int result = 0;
    for (int i = n - 1; i >= 0; --i)
        result = result << 1 | (output[i] > 0.5) ;//对输出结果四舍五入，并通过二进制转换为数
    return result;
}
class NN{
    Layer *bp;
    public:
    NN(NN &)=delete;
    void operator=(NN &)=delete;
    NN(double eta, double momentum, int layer[],int szLayer,ActionType actionType=SIGMOD){
      bp=CreateLayer(eta,momentum,layer,szLayer,actionType);
    }
    ~NN(){
      DestroyLayer(bp);
    }
    virtual int save(const char * path){
      return SaveLayer(bp,path);
    }
    void train(double input[], double target[]){
      bp->train(input,target);
    }
    void predict(double input[],double output[]){
      bp->predict(input,output);
    }
};
class RNN{
  Layer *bp;
  int * layer;
  double * input;
  int il,from;
  public:
  RNN(RNN &)=delete;
  void operator=(NN &)=delete;
  RNN(double eta, double momentum, int layer_i[],int szLayer,ActionType actionType=SIGMOD){
    this->layer=new int[szLayer];
    for(int i=0;i<szLayer;i++){
      layer[i]=layer_i[i];
    }
    from=szLayer-2;
    int f=layer[from];
    if(from<0)return;
    il=layer[0];
    layer[0]+=f;
    input=new double[layer[0]];
    bp = CreateLayer(eta,momentum,layer,szLayer,actionType);
  }
  RNN(const char * path){
    bp = LoadLayer(path);
    int szLayer=bp->szLayer;
    auto layer_i=bp->layer;
    this->layer=new int[szLayer];
    for(int i=0;i<szLayer;i++){
      layer[i]=layer_i[i];
    }
    from=szLayer-2;
    int f=layer[from];
    if(from<0)return;
    il=layer[0]-f;
    input=new double[layer[0]];
  }
  ~RNN(){
    if(!bp)return;
    DestroyLayer(bp);
    delete [] layer;
    delete [] input;
  }
  virtual int save(const char * path){
    return SaveLayer(bp,path);
  }
  inline void setInp(double inp[],bool re){
    int i;
    for(i=0;i<il;i++){
      input[i]=inp[i];
    }
    if(re){
      for(;i<layer[0];i++){
        input[i]=0;
      }
    }else{
      for(int j=0;i<layer[0];i++){
        input[i]=bp->output[from][j];
        j++;
      }
    }
  }
  virtual void train(double inp[], double target[],bool re=false){
    setInp(inp,re);
    bp->train(input,target);
  }
  virtual void predict(double inp[],double output[],bool re=false){
    setInp(inp,re);
    bp->predict(input,output);
  }
};
/////////////////////
}//namespace NN
}//namespace atulocher
#endif