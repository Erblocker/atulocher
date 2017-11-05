#ifndef atulocher_utils
#define atulocher_utils
#include <lua.hpp>
namespace atulocher{
  namespace utils{
    void luaopen(lua_State * L){
      
      luaL_openlibs(L);
    }
  }
}
#endif