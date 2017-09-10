#ifndef atulocher_GPU
#define atulocher_GPU
#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <sstream>
namespace atulocher{
    class GPU{
      public:
      cl_context context;
      cl_command_queue commandQueue;
      cl_device_id device;
      virtual void CreateContext(){
        cl_int errNum;
        cl_uint numPlatforms;
        cl_platform_id firstPlatformId;
        //选择可用的平台中的第一个
        errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
        if (errNum != CL_SUCCESS || numPlatforms <= 0){
          std::cerr << "Failed to find any OpenCL platforms." << std::endl;
          context=NULL;
          return;
        }
        //创建一个OpenCL上下文环境
        cl_context_properties contextProperties[] ={
          CL_CONTEXT_PLATFORM,
          (cl_context_properties)firstPlatformId,
          0
        };
        this->context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,
        NULL, NULL, &errNum);
      }
      //二、 创建设备并创建命令队列
      virtual void CreateCommandQueue(){
        cl_int errNum;
        cl_device_id *devices;
        size_t deviceBufferSize = -1;
        // 获取设备缓冲区大小
        errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
        if (deviceBufferSize <= 0){
          std::cerr << "No devices available.";
          return;
        }
        // 为设备分配缓存空间
        devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
        errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
        //选取可用设备中的第一个
        commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);
        this->device = devices[0];
        delete[] devices;
      }
      virtual cl_program CreateProgram(const char* srcStr){
        cl_int errNum;
        cl_program program;
        program = clCreateProgramWithSource(context, 1,
          (const char**)&srcStr,
            NULL, NULL);
        errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
        return program;
      }
      virtual cl_program CreateProgramWithPath(const char* fileName){
        cl_int errNum;
        cl_program program;
        std::ifstream kernelFile(fileName, std::ios::in);
        if (!kernelFile.is_open()){
          std::cerr << "Failed to open file for reading: " << fileName << std::endl;
          return NULL;
        }
        std::ostringstream oss;
        oss << kernelFile.rdbuf();
        std::string srcStdStr = oss.str();
        const char *srcStr = srcStdStr.c_str();
        program = clCreateProgramWithSource(context, 1,
          (const char**)&srcStr,
          NULL, NULL);
        errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
        return program;
      }
      virtual cl_kernel createKernel(cl_program program,const char * fn){
        return clCreateKernel(program, fn, NULL);
      }
      virtual void cleanup(){
        if (commandQueue != 0)
          clReleaseCommandQueue(commandQueue);
        if (context != 0)
          clReleaseContext(context);
      }
      virtual void initGPU(){
        CreateContext();
        CreateCommandQueue();
      }
    };
}
#endif