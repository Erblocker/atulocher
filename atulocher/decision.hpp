#ifndef atulocher_decision
#define atulocher_decision
#include "searchbuffer.hpp"
namespace atulocher{
    class decision:public searchbuffer{
        private:
            virtual void getActiveByPair(
                const char  *   data,
                const char  *   target,
                int             lastactivity,
                void(       *   callback)(int,int,void*),
                void        *   arg
            ){
                
            }
    };
    class trainer:public decision{
        public:
            virtual void train(){
                
            }
        
    };
}
#endif