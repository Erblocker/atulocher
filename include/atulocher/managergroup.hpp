#ifndef atulocher_mg
#define atulocher_mg
#include "NN.hpp"
namespace atulocher{
  namespace managerGroup{
    
    class Manager{
      NN::RNN * net;
      public:
      struct Outputbuffer{
        double dowhat         [512]; //做什么
        double arg            [4096];//参数
        double raiseInterest  [8];   //投票提升interest
                                     //  interest直接影响权重和继续的几率
        double after          [8];   //后续操作
                                     //  [0]结束，保留隐含层
                                     //  [1]结束，并且reset
                                     //  [2]重复，以arg为输入
                                     //  [3]重复，并且调用函数，获取结果
                                     //  [4]重复，以buffer为输入
                                     //  [5]重复，并且reset
                                     //  [6]调用函数，然后结束
                                     //  [7]设置buffer，然后重复
      }outputbuffer;
      struct Inputbuffer{
        double interest;
        double weight;
        double env[32];
        double arg[4096];
      }inputbuffer;
      
      double  weight;
      double  interest;
      
      Manager(const char * path){
        int ilen=sizeof(Outputbuffer)/sizeof(double);
        int olen=sizeof(Inputbuffer) /sizeof(double);
      }
      ~Manager(){
        
      }
      void train(){}
      void predict(){}
      
    };
    class ManGroup{
      public:
      struct{
        Manager::Outputbuffer buffer;
        int func;
        int mainly;
      }action;
      struct Member{
        Manager man;
      }mem[8];
      
      //events
      void(*onGetEvent)  (double*,ManGroup*);
      void(*onGetMain)   (double*,ManGroup*);
      void(*onDoActivity)(double*,ManGroup*);
      bool(*onCallFunc)  (double*,ManGroup*);
      
      void setArg(double * a){
        
      }
      void setEnv(double * a){
        
      }
      void think(){
        
      }
      int getMain(){
        double max=mem[0].man.interest*mem[0].man.weight;
        int maxi=0;
        for(int i=1;i<8;i++){
          double pn=mem[i].man.interest*mem[i].man.weight;
          if(pn>max){
            max=pn;
            maxi=i;
          }
        }
        return maxi;
      }
      void updateInterest(){}
      void updateOut(){
        action.mainly=getMain();
        
        for(int i=0;i<512;i++){
          action.buffer.dowhat[i]=mem[action.mainly].man.outputbuffer.dowhat[i];
        }
        
        for(int i=0;i<4096;i++){
          action.buffer.arg[i]=mem[action.mainly].man.outputbuffer.arg[i];
        }
        
        for(int i=0;i<8;i++){
          double sr=0;
          double sa=0;
          for(int j=0;j<8;j++){
            sr+=mem[j].man.outputbuffer.raiseInterest[i]*mem[j].man.weight*mem[j].man.interest;
            sa+=mem[j].man.outputbuffer.after[i]*mem[j].man.weight*mem[j].man.interest;
          }
          action.buffer.raiseInterest[i]=sr;
          action.buffer.after[i]=sa;
        }
        
        for(int i=0;i<8;i++){
          mem[i].man.interest*=0.7;
        }
      }
      void doActivity(){
        
      }
      void run(){
        
      }
    };
  }
}
#endif