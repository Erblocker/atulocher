#ifndef atulocher_lstm
#define atulocher_lstm
#include <stdio.h>
#include <cblas.h>
#include <math.h>
namespace atulocher{
  namespace lstm{
    //  from:blas_ext.c
    //  project:rnn_test
    //  Created by xuzhuoyi on 2016/12/30.
    //  Modify  by cgoxopx  on 2017/12/30.
    void gemm(
          const enum CBLAS_TRANSPOSE TransA,
          const enum CBLAS_TRANSPOSE TransB,
          const int M,
          const int N,
          const int K,
          const double alpha,
          const double *A,
          const double *B,
          const double beta,
          double *C
    ){
      int lda = (TransA == CblasNoTrans)?K:M;
      int ldb = (TransB == CblasNoTrans)?N:K;
      int ldc = N;
      
      cblas_dgemm(CblasRowMajor,TransA,TransB,M,N,K,alpha,A,lda,B,ldb,beta,C,ldc);
      
    }
    void copy(const int N, const double *x, double *y){
      cblas_dcopy(N, x, 1, y, 1);
    }
    void dcopy(const int N, const double *x, double *y){
      cblas_dcopy(N,x,1,y,1);
    }
    double sigmoid(double val){
      return 1.0/(1.0+exp(-val));
    }
    double _tanh(double val){
      return (1.0-exp(-2.0*val))/(1.0+exp(-2.0*val));
    }
    double ReLU(double val){
      return (val>0)?val:0;
    }
    void copy_n_rows(const int rows, const int N, const double *x, double *A){
      int i = 0;
      for(i = 0;i < rows; i++){
        dcopy(N, x, A);
        A += N;
      }
    }
    void activation(double *pData,int length,double (*non_linear)(double)){
      int i = 0;
      double *pTmp = pData;
      for(i = 0; i<length;i++){
        *pData++ = non_linear(*pTmp++);
      }
    }
    double add(double a,double b){
      return a+b;
    }
    double sub(double a,double b){
      return a-b;
    }
    double mul(double a,double b){
      return a*b;
    }
    void tanh_batch(double *A,double *B,int N){
      int i = 0;
      for(i = 0;i<N;i++){
        *B++ += _tanh(*A++);
      }
    }
    void elementWise(double *A,double *B,double *C,int N,double (*op)(double,double)){
      int i = 0;
      for(i = 0;i<N;i++){
          double val1 = *A++;  //equal then increment
          double val2 = *B++;  //equal then increment
          *C++ += op(val1,val2);
      }
    }
    double dsigmoid(double val){
      return val*(1.0-val);
    }
    double dtanh(double val){
      return 1.0 - val*val;
    }
    double dReLU(double val){
      return (val>0)?1:0;
    }
    void dactivation(double *pSrcData,double *pDstData,int length, double(*derivation)(double)){
      int i = 0;
      for(i = 0;i<length;i++){
        *pDstData++ *= derivation(*pSrcData++);
      }
    }
    void axpy(const int N, const double alpha, const double *x, double *y){
      cblas_daxpy(N,alpha,x,1,y,1);
    }
    void add_n_rows(const int rows, const int N, const double *x, double *A){
      int i = 0;
      for(i = 0; i< rows;i++){
        axpy(N,1.0,x,A);
        x += N;
      }
    }
    double nrm2(const int N, const double *x){
      return cblas_dnrm2(N,x,1);
    }
    struct Layer{
      double *pW;
      double *pU;
      double *pb;
      double *ph;
      double *pInput;
      double *pForget;
      double *pCell_h;  //cell hat
      double *pOutput;
      double *pCell;
      double *pCell_t; //cell t-1
      int nWs;
      int nUs;
      int nbs;
      int nhs;
      int nNodes;
      int nPrevs;
      int *pPrev;
    };
    struct Nets{
      double *pWeight;
      double *pBias;
      Layer  *pLayer;
      int nLayers;
      int nSamples;
      int nMaxSeqs;
      int nFeatures;  //input dimension??
      int nWeights;
      int nBias;
    };
    void lstm_forward(
      double *x,
      double *h,
      double *input,
      double *output,
      double *forget,
      double *cell,
      double *cell_hat,
      double *cell_tanh,
      double *W,
      double *U,
      double *b,
      int nSamples,
      int nSeqs,
      int nFeatures,
      int nHiddens,
      bool isInputLayer
    ){
      int M = nSeqs*nSamples;  //not include nhiddens,nseqs is the outside dimension
    
      int b_offset = nHiddens;
      int W_offset = nHiddens*nHiddens;
      int U_offset = nHiddens*nFeatures;
//    int h_offset = nHiddens;
//    int x_offset = nSamples*nSeqs*nFeatures;
    
      int N = nSamples*nHiddens;  //not include nSeqs
    
      int i = 0;
    
      double *tmpB = b;
      double *tmpU = U;
      double *tmpW = W;
      double *tmpH = h;
      double *tmpC = cell;
    
            //****************t = 0
            //plus bias
            copy_n_rows(M,nHiddens,tmpB,input);
            tmpB += b_offset;
            
            copy_n_rows(M,nHiddens,tmpB,forget);
            tmpB += b_offset;
            
            copy_n_rows(M,nHiddens,tmpB,output);
            tmpB += b_offset;
            
            copy_n_rows(M,nHiddens,tmpB,cell_hat);
            
            //plus Ux
            gemm(CblasNoTrans,CblasTrans,M,nHiddens,nFeatures,1.0,x,tmpU,1.0,input); //C = A*B+C input = input+tmpU*nFeatures
            
            tmpU += U_offset;
            gemm(CblasNoTrans,CblasTrans,M,nHiddens,nFeatures,1.0,x,tmpU,1.0,forget); //forget
            
            tmpU += U_offset;
            gemm(CblasNoTrans,CblasTrans,M,nHiddens,nFeatures,1.0,x,tmpU,1.0,output); //output
            
            tmpU += U_offset;
            gemm(CblasNoTrans,CblasTrans,M,nHiddens,nFeatures,1.0,x,tmpU,1.0,cell_hat); //C~
            
            activation(input,N,sigmoid);
            activation(forget,N,sigmoid);
            activation(output,N,sigmoid);
            activation(cell_hat,N,_tanh);
            
            //Ct = ft*Ct-1 + it*Chat-t
            elementWise(input, cell_hat, cell, N, mul);  //where is cellt-1
            
            //tanh(Ct)
            tanh_batch(cell, cell_tanh, N);
            //h = ot*tanh(Ct)
            elementWise(output, cell_tanh, h, N, mul);
            //这里的x是所有的输入的序列长度，nSeqs*nSamples*nFeatures,下面这个移位只是移动了一个timestep的长度！！注意！！！并不是移动了所有的seqs
            h += N;
            input += N;
            forget += N;
            output += N;
            cell_hat += N;
            cell_tanh += N;
            cell += N;
            //***********begin t = 1 to the end
            //
            for(i = 1;i <nSeqs;i++){
                //参数夸时刻共享
                tmpW = W;  //every element in the sequence have the same weights
                gemm(CblasNoTrans, CblasTrans, nSamples, nHiddens, nHiddens, 1.0, tmpH, tmpW, 1.0, input);
                tmpW += W_offset;
                gemm(CblasNoTrans, CblasTrans, nSamples, nHiddens, nHiddens, 1.0, tmpH, tmpW, 1.0, forget);
                tmpW += W_offset;
                gemm(CblasNoTrans, CblasTrans, nSamples, nHiddens, nHiddens, 1.0, tmpH, tmpW, 1.0, output);
                tmpW += W_offset;
                gemm(CblasNoTrans, CblasTrans, nSamples, nHiddens, nHiddens, 1.0, tmpH, tmpW, 1.0, cell_hat);
                
                activation(input, N, sigmoid);
                activation(forget, N, sigmoid);
                activation(output, N, sigmoid);
                activation(cell_hat, N, _tanh);
                
                elementWise(input, cell_hat, cell, N, mul);
                elementWise(forget, tmpC, cell, N, mul);
                tanh_batch(cell, cell_tanh, N);
                elementWise(output, cell_tanh, h, N, mul);
                tmpH += N;
                tmpC += N;
                h += N;
                input += N;
                forget += N;
                output += N;
                cell_hat += N;
                cell_tanh += N;
                cell += N;
            }
    }
    void lstm_backward(
      double *dEdh,
      double *dEdct,
      double *dEdo,
      double *dEdi,
      double *dEdf,
      double *dEdc,
      double *dEdch,
      double *dEdx,
      double *dEdWi,
      double *dEdUi,
      double *dEdWf,
      double *dEdUf,
      double *dEdWo,
      double *dEdUo,
      double *dEdWc,
      double *dEdUc,
      double *dEdbi,
      double *dEdbf,
      double *dEdbo,
      double *dEdbc,
      double *x,
      double *input,
      double *forget,
      double *output,
      double *cell,
      double *cell_hat,
      double *cell_tanh,
      double *h,
      double *Wi,
      double *Ui,
      double *Wf,
      double *Uf,
      double *Wo,
      double *Uo,
      double *Wc,
      double *Uc,
      int nSeqs,
      int nSamples,
      int nHiddens,
      int nFeatures,
      bool isInput
    ){
      int M = nSeqs*nSamples;
      int b_offset = 4*nHiddens;
      int W_offset = 4*nHiddens*nHiddens;
      int U_offset = (isInput?4:8)*nHiddens*nFeatures;
      int h_offset = M*nHiddens;
      int x_offset = M*nFeatures;
    
      int N = nSamples*nHiddens;
      int L = nSamples*nFeatures;
    
      int i = 0;
    
      double *tmpC = cell;
      double *tmpH = h;
      double *tmpDc = dEdc;
      double *tmpDh = dEdh;
      
      double *tmpCell = cell;
      double *tmpCell_tanh = cell_tanh;
      double *tmpCell_hat = cell_hat;
      double *tmpForget = forget;
      double *tmpOutput = output;
      double *tmpHH = h;
      double *tmpXX = x;
    
      double *tmpdEdh = dEdh;
      double *tmpdEdo = dEdo;
      double *tmpdEdct = dEdct;
      double *tmpdEdi = dEdi;
      double *tmpdEdch = dEdch;
      double *tmpdEdf = dEdf;
      double *tmpdEdx = dEdx;
    
            //first point to the end of vector move to last timestep, then go to front
            cell += h_offset -N;
            cell_tanh += h_offset -N;
            cell_hat += h_offset -N;
            forget += h_offset -N;
            output += h_offset -N;
            input += h_offset -N;
            h += h_offset -N;
            x += x_offset -L;
            
            dEdh += h_offset -N;
            dEdo += h_offset -N;
            dEdc += h_offset -N;
            dEdct += h_offset -N;
            dEdi += h_offset -N;
            dEdch += h_offset -N;
            dEdf += h_offset -N;
            
            if(!isInput){
                dEdx += x_offset -L;
            }
            
            tmpDc = dEdc - N; //one timestep before
            tmpC = cell - N;
            tmpH = h - N;
            tmpDh = dEdh - N;
            
            for(i = 0;i<nSeqs;i++){
                elementWise(dEdh, cell_tanh, dEdo, N, mul);  //dEdOt
                elementWise(dEdh, output, dEdct, N, mul);  //dEdCt * Ot
                dactivation(cell_tanh, dEdct, N, dtanh); //dEdCt*Ot  * dtanh(Ct)
                
                axpy(N, 1.0, dEdct, dEdc);  //dEdC = dEdC + dEdCt
                elementWise(dEdc, cell_hat, dEdi, N, mul); //dEdi = dEdC * cell_hat
                elementWise(dEdc, input, dEdch, N, mul);  //dEdChat = input * dEdC
                
                if(i < nSeqs - 1){  //except t = 0
                    elementWise(dEdc, forget, tmpDc, N, mul);  //not understand yet
                    elementWise(dEdc, tmpC, dEdf, N, mul);
                }
                dactivation(input, dEdi, N, dsigmoid);  //dEdi = dsig(it) for the preparation for wi and ui
                if(i < nSeqs - 1){
                    dactivation(forget, dEdf, N, dsigmoid);  //dEdf = dsig(ft)
                }
                dactivation(output, dEdo, N, dsigmoid);  //dEdo = dsig(Ot)
                dactivation(cell_hat, dEdch, N, dtanh); //dEdch(tanh) = dsig(cell_hat)
                
                if(i < nSeqs - 1){
                    gemm(CblasTrans, CblasNoTrans, nHiddens, nHiddens, nSamples, 1.0, dEdi, tmpH, 1.0, dEdWi);  //it = sig(wi*ht-1 + ui*xt + bi)
                    gemm(CblasTrans, CblasNoTrans, nHiddens, nHiddens, nSamples, 1.0, dEdf, tmpH, 1.0, dEdWf);
                    gemm(CblasTrans, CblasNoTrans, nHiddens, nHiddens, nSamples, 1.0, dEdo, tmpH, 1.0, dEdWo);
                    gemm(CblasTrans, CblasNoTrans, nHiddens, nHiddens, nSamples, 1.0, dEdch, tmpH, 1.0, dEdWc);
                }
                if(i < nSeqs - 1){
                    gemm(CblasNoTrans, CblasNoTrans, nSamples, nHiddens, nHiddens, 1.0, dEdi, Wi, 1.0, dEdh);
                    gemm(CblasNoTrans, CblasNoTrans, nSamples, nHiddens, nHiddens, 1.0, dEdf, Wf, 1.0, dEdh);
                    gemm(CblasNoTrans, CblasNoTrans, nSamples, nHiddens, nHiddens, 1.0, dEdo, Wo, 1.0, dEdh);
                    gemm(CblasNoTrans, CblasNoTrans, nSamples, nHiddens, nHiddens, 1.0, dEdch, Wc, 1.0, dEdh);
                }
                gemm(CblasTrans, CblasNoTrans, nHiddens, nFeatures, nSamples, 1.0, dEdi, x, 1.0, dEdUi);
                if(i < nSeqs - 1){
                    gemm(CblasTrans, CblasNoTrans, nHiddens, nFeatures, nSamples, 1.0, dEdf, x, 1.0, dEdUf);
                }
                gemm(CblasTrans, CblasNoTrans, nHiddens, nFeatures, nSamples, 1.0, dEdo, x, 1.0, dEdUo);
                gemm(CblasTrans, CblasNoTrans, nHiddens, nFeatures, nSamples, 1.0, dEdch, x, 1.0, dEdUc);
                
                if(!isInput){ //dEdx is cancha theta
                    gemm(CblasNoTrans, CblasNoTrans, nSamples, nFeatures, nHiddens, 1.0, dEdi, Ui, 1.0,dEdx);
                    if(i < nSeqs - 1){
                        gemm(CblasNoTrans, CblasNoTrans, nSamples, nFeatures, nHiddens, 1.0, dEdf, Uf, 1.0, dEdx);
                    }
                    gemm(CblasNoTrans, CblasNoTrans, nSamples, nFeatures, nHiddens, 1.0, dEdo, Uo, 1.0, dEdx);
                    gemm(CblasNoTrans, CblasNoTrans, nSamples, nFeatures, nHiddens, 1.0, dEdch, Uc, 1.0, dEdx);
                }
                add_n_rows(nSamples, nHiddens, dEdi, dEdbi);
                if(i < nSeqs - 1){
                    add_n_rows(nSamples, nHiddens, dEdf, dEdbf);
                }
                add_n_rows(nSamples, nHiddens, dEdo, dEdbo);
                add_n_rows(nSamples, nHiddens, dEdch, dEdbc);
                
                cell_tanh -= N;
                cell_hat -= N;
                cell -= N;
                input -= N;
                forget -= N;
                input -= N;
                output -= N;
                h -= N;
                x -= L;  //if is input layer?
                
                dEdh -= N;
                dEdo -= N;
                dEdf -= N;
                dEdc -= N;
                dEdct -= N;
                dEdch -= N;
                dEdi -= N;
                if(!isInput){
                    dEdx -= L;
                }
                if(i < nSeqs - 1){ //not t=0
                    tmpDc -= N;
                    tmpC -= N;
                    tmpH -= N;
                    tmpDh -= N;
                }
            }
    }
    double delta_loss(
      double *dEdy,
      double *dEdW,
      double *dEdb,
      double *dEdh,
      double *y,
      double *label,
      double *h,
      double *W,
      int nSeqs,
      int nSamples,
      int nHiddens,
      int nYs,
      int isInput
    ){
      int offset1 = nSamples*nYs;  //nys output
      //int offset = nSamples*nHiddens;
      double *tmpH = h;
      double *tmpDy = dEdy;
      double cost = 0;
      double val = 0;
    
            copy(nSeqs*offset1, y, dEdy); //copy y to dEdy, y is output
            axpy(nSeqs*offset1,-1.0, label, dEdy);  //dEdy = y -label
            val = nrm2(nSeqs*offset1, dEdy); //normalize
            cost += val*val/2.0;  //(y-label)^2 * 0.5
            dactivation(y, dEdy, nSeqs*offset1, dsigmoid); //dEdy
            gemm(CblasTrans, CblasNoTrans, nYs, nHiddens, nSeqs*nSamples, 1.0, tmpDy, tmpH, 1.0, dEdW); //dEdW += Dy * ht
            add_n_rows(nSeqs*nSamples, nYs, tmpDy, dEdb);  //dEdb += dEdy
            if(!isInput){
                gemm(CblasNoTrans, CblasNoTrans, nSeqs*nSamples, nHiddens, nYs, 1.0, dEdy, W, 1.0, dEdh);
            }
      return cost;
    }
    void forward(double *pFeat,Nets *pNet,int nSamples,int nSeqs){
      int i = 0;
      Layer *pLayer = pNet->pLayer;
      for(i = 0;i<pNet->nLayers;i++){
        if(pLayer->pPrev[0] == -1){
          lstm_forward(pFeat, pLayer->ph, pLayer->pInput, pLayer->pOutput, pLayer->pForget, pLayer->pCell, pLayer->pCell_h, pLayer->pCell_t, pLayer->pW, pLayer->pU, pLayer->pb,nSamples, nSeqs, pNet->nFeatures, pLayer->nNodes,1);
        }else{
          lstm_forward(pNet->pLayer[pLayer->pPrev[0]].ph, pLayer->ph, pLayer->pInput, pLayer->pOutput, pLayer->pForget, pLayer->pCell, pLayer->pCell_h, pLayer->pCell_t, pLayer->pW, pLayer->pU, pLayer->pb,nSamples,nSeqs, pNet->pLayer[pLayer->pPrev[0]].nNodes, pLayer->nNodes,0);
        }
        pLayer++;
      }
    }
  }
}
#endif