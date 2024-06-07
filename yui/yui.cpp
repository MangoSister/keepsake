#include "yui.h"

namespace ks
{

struct Foo
{
    int x = 0;

    sol::table generate_metadata() const { return sol::table(); }

    static std::shared_ptr<Foo> load_from_metadata(const sol::table &metadata) { return nullptr; }
};

struct Bar
{
    Bar(int a, const Foo &foo) : a(a), foo(&foo) {}

    int a = 0;
    const Foo *foo = nullptr;
};

void foo()
{
    yui::AssetTable asset_table;

    std::shared_ptr<Foo> foo;

    yui::AssetRef<Foo> foo_ref = asset_table.add(foo);
    foo_ref.clear();
    if (asset_table.lookup(foo_ref)) {
    }

    yui::AssetRef<Foo> ref1;
    int y = ref1.ptr->x;
    y = ref1->x;

    sol::table a;
    sol::table b = a;

    sol::state lua;
    lua.open_libraries(sol::lib::base, sol::lib::jit);

    sol::usertype<Foo> foo_type = lua.new_usertype<Foo>("Foo", sol::constructors<Foo()>());
    foo_type["x"] = &Foo::x;

    sol::usertype<Bar> bar_type = lua.new_usertype<Bar>("Bar", sol::constructors<Bar(int, const Foo &)>());
    bar_type["a"] = &Bar::a;
    bar_type["foo"] = &Bar::foo;

    try {
        lua.safe_script(R"(
            print(jit.version)
	        obj = nil
            f = Foo.new()
            b = Bar.new(123, f)
            print(b.a)
            print(b.foo))");

    } catch (const sol::error &e) {
        std::cerr << "Lua script error: " << e.what() << std::endl;
        std::exit(1);
    }
    // try {
    //     auto result1 = lua.safe_script_file("bad.code");
    // } catch (const sol::error &e) {
    //     std::cerr << "Lua script error: " << e.what() << std::endl;
    //     std::exit(1);
    // }
    //  lua["a"].
}

} // namespace ks

// -- scriptable front end
// -- - asset loading
// -- - task 
// -- - simple "gameplay"/animation
// -----------------------------
// require "more_asset.lua"

// y = yui.get_inst()

// brick = y.create_texture{
//     name = "brick",
//     path = "brick.png", 
//     colorspace = sRGB,
//     channels = 3,
//     build_mipmaps = false
// }

// brick_bsdf = y.create_bsdf{
//     type = "principled_brdf", 
//     basecolor = ShaderField{
//         type = "texture",
//         map = brick,
//     }}

// metal_bsdf =  y.create_bsdf{
//     type = "principled_brdf", 
//     basecolor = {
//         type = "texture",
//         map = "texture.brick",
//     }}

// y.create_some_custom_stuff{}

// --------------------------------

// for i in range(10):
//     camera.pos.x += 2
//     y.small_pt{camera}