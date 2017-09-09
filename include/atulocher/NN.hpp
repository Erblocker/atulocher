#ifndef atulocher_NN
#define atulocher_NN
#include "NN_MCPU.hpp"
#include <mutex>
#include <condition_variable>
#include <atomic>
#include "threadpool.hpp"
#include "mempool.hpp"
namespace atulocher{
namespace NN{
int SaveLayer(Layer *pLayer,const char *filename){
    if(!pLayer) return 0;
    FILE *fp =0;
    if((fp = fopen(filename, "wb+"))){
        fprintf(fp,"atulocher ann std\n");
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
        bzero(buf,4096);
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
    Layer* pLayer = new Layer_MCPU(eta,momentum,layer,szLayer,actionType);
    return pLayer;
}
 
int DestroyLayer(Layer *pLayer){
    if (!pLayer) return 0;
    delete pLayer;
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