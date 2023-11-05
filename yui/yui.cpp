#include "yui.h"
#define SOL_ALL_SAFETIES_ON 1
#include <sol/sol.hpp>

namespace ks
{

void foo()
{
    sol::state lua;
    lua.open_libraries(sol::lib::base, sol::lib::jit);
    lua.script("print(jit.version)");
}

} // namespace ks