#ifndef atulocher_utils
#define atulocher_utils
#include <lua.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <sys/file.h>
#include <sys/types.h>
namespace atulocher{
  namespace utils{
    void luaopen(lua_State * L){
      
      luaL_openlibs(L);
    }
  }
}
#endif