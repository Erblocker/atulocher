#ifndef atulocher_NN
#define atulocher_NN
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <memory.h>
#include <unistd.h>

namespace atulocher{
namespace NN{
typedef enum {CUSTOM,SIGMOD} ActionType;
typedef double(*Function)(double);
typedef struct {
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
}BPAnn;
 
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
 
static int GetCbSize(int szLayer,int layer[]){
    int cbSize = sizeof(BPAnn);
    cbSize += sizeof(int)*szLayer;
    cbSize += sizeof(double *)*(szLayer*6-5);
    cbSize += sizeof(double)*layer[0];
    for (int i = 1; i < szLayer; ++i){
        cbSize += sizeof(double)*layer[i] * layer[i - 1]*2;
        cbSize += sizeof(double)*layer[i]*4;
    }
    return cbSize;
}
 
void InitBPAnn(void *buffer){
    BPAnn *pBPAnn = (BPAnn *)buffer;
    switch (pBPAnn->actionType) {
        case SIGMOD:
            pBPAnn->act=Sigmod;
            pBPAnn->actdiff = SigmodDiff;
            break;
             
        default:
            pBPAnn->act = 0;
            pBPAnn->actdiff = 0;
            break;
    }
     
    int szLayer = pBPAnn->szLayer;
    pBPAnn->layer = (int *)pBPAnn->buffer;
    pBPAnn->output = (double **)(pBPAnn->layer+szLayer);
    pBPAnn->delta = pBPAnn->output+szLayer;
    pBPAnn->weights = pBPAnn->delta + szLayer - 1;
    pBPAnn->theta = pBPAnn->weights + szLayer - 1;
    pBPAnn->preWeights = pBPAnn->theta + szLayer - 1;
    pBPAnn->preTheta = pBPAnn->preWeights + szLayer - 1;
     
    *(pBPAnn->output) = (double *)(pBPAnn->preTheta + szLayer - 1);
    int *layer = pBPAnn->layer;
    for (int i = 0; i < szLayer; ++i)
        pBPAnn->output[i+1]=pBPAnn->output[i]+layer[i];
    for (int i=0;i<szLayer - 1;++i)
        pBPAnn->delta[i+1]=pBPAnn->delta[i]+layer[i+1];
    for (int i = 0; i < szLayer - 1; ++i)
        pBPAnn->weights[i+1]=pBPAnn->weights[i]+layer[i] * layer[i + 1];
    for(int i=0;i<szLayer - 1;++i)
        pBPAnn->theta[i+1]=pBPAnn->theta[i]+layer[i+1];
    long long tmp = pBPAnn->theta[szLayer - 1]-pBPAnn->weights[0];
    for(int i=0;i<szLayer - 1;++i){
        pBPAnn->preWeights[i]=pBPAnn->weights[i]+tmp;
        pBPAnn->preTheta[i] = pBPAnn->theta[i]+tmp;
    }
}
 
int SaveBPAnn(BPAnn *pBPAnn,const char *filename){
    if(!pBPAnn) return 0;
    FILE *fp =0;
    if((fp = fopen(filename, "wb+"))){
        fwrite(pBPAnn, 1, pBPAnn->cbSize, fp);
        return fclose(fp);
    }
    return 0;
}
 
BPAnn* LoadBPAnn(const char *filename){
    if(!filename) return 0;
    FILE *fp = 0;
    if((fp=fopen(filename, "rb+"))){
        int szfile = 0;
        fread(&szfile, 4, 1, fp);
        fseek(fp, 0, SEEK_SET);
        BPAnn *pBPAnn = (BPAnn *)malloc(szfile);
        if(fread(pBPAnn, 1, szfile, fp)<szfile) return 0;
        InitBPAnn(pBPAnn);
        return pBPAnn;
    }
    return 0;
}
 
BPAnn* CreateBPAnn(double eta, double momentum, int layer[],int szLayer, ActionType actionType){
    int cbSize = GetCbSize(szLayer, layer);
    BPAnn* pBPAnn = (BPAnn *)malloc(cbSize);
    pBPAnn->cbSize = cbSize;
    pBPAnn->eta = eta;
    pBPAnn->momentum = momentum;
    pBPAnn->szLayer = szLayer;
    pBPAnn->actionType = actionType;
    pBPAnn->layer = (int *)pBPAnn->buffer;
    for(int i=0;i<szLayer;++i)
        pBPAnn->layer[i] = layer[i];
    InitBPAnn(pBPAnn);
    for(double *i=pBPAnn->weights[0];i!=pBPAnn->preWeights[0];++i)
        *i=lfrand();
    for(double *i=pBPAnn->preWeights[0];i!=(double *)((unsigned char *)pBPAnn+cbSize);++i)
        *i=0;
    return pBPAnn;
}
 
int DestroyBPAnn(BPAnn *pBPAnn){
    if (!pBPAnn) return 0;
    free(pBPAnn);
    return 1;
}
 
static void LoadInput(double input[],BPAnn *pBPAnn){
    for (int i = 0; i < pBPAnn->layer[0]; ++i)
        pBPAnn->output[0][i] = input[i];
}
 
static void LoadTarget(double target[], BPAnn *pBPAnn){
    int lastIndex = pBPAnn->szLayer - 1;
    double *delta = pBPAnn->delta[lastIndex - 1];
    double *output = pBPAnn->output[lastIndex];
    for (int i = 0; i < pBPAnn->layer[lastIndex]; ++i)
        delta[i] = pBPAnn->actdiff(output[i])*(target[i] - output[i]);
}
 
static void Forward(BPAnn *pBPAnn){
    int lastIndex = pBPAnn->szLayer - 1;
    int *layer = pBPAnn->layer;
    double **weights = pBPAnn->weights;
    double **output = pBPAnn->output;
    double **theta = pBPAnn->theta;
    Function act = pBPAnn->act;
    for (int i = 0; i < lastIndex; ++i){
        MatXMat(output[i], weights[i], output[i + 1], 1, layer[i + 1], layer[i]);
        for (int j = 0; j < layer[i + 1]; ++j)
            output[i + 1][j] = act(output[i + 1][j] + theta[i][j]);
    }
}
 
static void CalculateDelta(BPAnn *pBPAnn){
    int lastIndex = pBPAnn->szLayer - 1;
    int *layer = pBPAnn->layer;
    double **weights = pBPAnn->weights;
    double **output = pBPAnn->output;
    double **delta = pBPAnn->delta;
    Function actdiff = pBPAnn->actdiff;
    for (int i = lastIndex-1; i > 0; --i){
        MatXMat(weights[i], delta[i], delta[i - 1], layer[i], 1, layer[i + 1]);
        for (int j = 0; j < layer[i]; ++j)
            delta[i - 1][j] *= actdiff(output[i][j]);
    }
}
 
static void AdjustWeights(BPAnn *pBPAnn){
    int lastIndex = pBPAnn->szLayer - 1;
    int *layer = pBPAnn->layer;
    double **weights = pBPAnn->weights;
    double **output = pBPAnn->output;
    double **delta = pBPAnn->delta;
    double **preWeights = pBPAnn->preWeights;
    double **theta = pBPAnn->theta;
    double **preTheta = pBPAnn->preTheta;
    double momentum = pBPAnn->momentum;
    double eta = pBPAnn->eta;
    for (int i = 0; i < lastIndex; ++i){
        for (int j = 0; j < layer[i]; ++j)
            for (int k = 0; k < layer[i + 1]; ++k){
                int pos = j*layer[i + 1] + k;
                preWeights[i][pos] = momentum * preWeights[i][pos] + eta * delta[i][k] * output[i][j];
                weights[i][pos] += preWeights[i][pos];
            }
         
        for (int j = 0; j < layer[i + 1]; ++j){
            preTheta[i][j] = momentum*preTheta[i][j] + eta*delta[i][j];
            theta[i][j] += preTheta[i][j];
        }
    }
}
 
void Train(double input[], double target[],BPAnn *pBPAnn){
    if(pBPAnn->act&&pBPAnn->actdiff){
        LoadInput(input, pBPAnn);
        Forward(pBPAnn);
        LoadTarget(target,pBPAnn);
        CalculateDelta(pBPAnn);
        AdjustWeights(pBPAnn);
    }
}
 
void Predict(double input[],double output[],BPAnn *pBPAnn){
    if(pBPAnn->act&&pBPAnn->actdiff){
        int lastIndex = pBPAnn->szLayer - 1;
        LoadInput(input, pBPAnn);
        Forward(pBPAnn);
        double *result = pBPAnn->output[lastIndex];
        for (int i = 0; i < pBPAnn->layer[lastIndex]; ++i)
            output[i] = result[i];
    }
}
 
 
 
static void ToBinary(unsigned x, unsigned n,double output[]){
    for (unsigned i = 0, j = x; i < n; ++i, j >>= 1)
        output[i] = j & 1;
}
 
static unsigned FromBinary(double output[],unsigned n){
    int result = 0;
    for (int i = n - 1; i >= 0; --i)
        result = result << 1 | (output[i] > 0.5) ;//对输出结果四舍五入，并通过二进制转换为数
    return result;
}
 
//使用神经网络进行异或运算，输入为2个0~32767之间的数，前15节点为第1个数二进制，后15节点为第2个数二进制，输出为结果的二进制
static void TestXor(){
    int layer[] = { 30,48,15 };
    BPAnn *bp = CreateBPAnn(0.25, 0.9, layer, 3, SIGMOD);
    double input[30], output[15];
    int error = 0;
    for (int j = 0; j < 100; ++j){
        for (int i = 0; i < 400; ++i){
            unsigned x = rand() & 32767;
            unsigned y = rand() & 32767;
            ToBinary(x << 15 | y, 30, input);
            ToBinary(x^y, 15, output);
            Train(input, output, bp);
        }
        printf("%02d%%\n", j + 1);
    }
    printf("\nfinish train\n");
    for (int i = 0; i < 2000; ++i){
        unsigned x = rand() & 32767;
        unsigned y = rand() & 32767;
        ToBinary(x << 15 | y, 30, input);
        Predict(input, output, bp);
        unsigned result = FromBinary(output, 15);
        if(result != (x^y)){
            ++error;
            printf("%u^%u=%u\tpredict=%u\n", x, y, x^y, result);
        }
    }
    printf("error = %d\n", error);
    SaveBPAnn(bp,"BP.ann");
    DestroyBPAnn(bp);
}
class NN{
    BPAnn *bp;
    public:
    NN(NN &)=delete;
    void operator=(NN &)=delete;
    NN(double eta, double momentum, int layer[],int szLayer,ActionType actionType){
        bp = CreateBPAnn(eta,momentum,layer,szLayer,actionType);
    }
    NN(const char * path){
        LoadBPAnn(path);
    }
    ~NN(){
        DestroyBPAnn(bp);
    }
    int save(const char * path){
        return SaveBPAnn(bp,path);
    }
    void train(double input[], double target[]){
      Train(input,target,bp);
    }
    void predict(double input[],double output[]){
      Predict(input,output,bp);
    }
};
/////////////////////
}//namespace NN
}//namespace atulocher
#endif