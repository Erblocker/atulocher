#ifndef yrssf_atulocher_module
#define yrssf_atulocher_module
#include "luapool.hpp"
namespace atulocher{
  class module{
    luapool::luap * lp;
    public:
    lua_State * L;
    module(){
      lp=luapool::Create();
      L=lp->L;
    }
    ~module(){
      luapool::Delete(lp);
    }
    virtual void run(){
      lua_getglobal(L,"main");
      lua_pcall(L,1,1,0);
      lua_pop(L,1);
    }
  };
}
#endif