#ifndef atulocher_active
#define atulocher_active
#include "dectree.hpp"
#include "language.hpp"
namespace atulocher{
  class active{
    public:
    dectree::dectree dect;
    lang  * lsolv;
    std::string argstr;
    std::list<std::string> res;
    virtual void run(){
      lsolv->solve(argstr);
      for(auto it:lsolv->keymeans){
        try{
          set<string> dep;
          dect.addknown(it.first,it.second,it.second,dep);
        }catch(dectree::NodeExist){
          
        }
      }
      dect.compute();
      dect.printTree(res);
    }
  };
}
#endif