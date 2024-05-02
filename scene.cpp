#include "scene.h"
#include "mesh_asset.h"
#include "normal_map.h"

namespace ks
{

SubScene::~SubScene()
{
    if (rtcscene) {
        rtcReleaseScene(rtcscene);
        rtcscene = nullptr;
    }
}

SubScene::SubScene(SubScene &&other)
{
    geometries = std::move(other.geometries);
    materials = std::move(other.materials);
    rtcscene = other.rtcscene;
    // Avoid releasing...
    other.rtcscene = nullptr;
}

SubScene &SubScene::operator=(SubScene &&other)
{
    if (this == &other) {
        return *this;
    }

    if (rtcscene) {
        rtcReleaseScene(rtcscene);
        rtcscene = nullptr;
    }

    geometries = std::move(other.geometries);
    materials = std::move(other.materials);
    rtcscene = other.rtcscene;
    // Avoid releasing...
    other.rtcscene = nullptr;
    return *this;
}

void SubScene::create_rtc_scene(const EmbreeDevice &device)
{
    rtcscene = rtcNewScene(device);
    // Later useful for different custom precomputation operations.
    rtcSetSceneFlags(rtcscene, RTC_SCENE_FLAG_CONTEXT_FILTER_FUNCTION);

    for (int i = 0; i < (int)geometries.size(); ++i) {
        Geometry &geom = *geometries[i];
        if (!geom.rtcgeom)
            geom.create_rtc_geom(device);
        rtcAttachGeometry(rtcscene, geom.rtcgeom);
    }

    // build bvh, etc.
    rtcCommitScene(rtcscene);
}

Scene::~Scene()
{
    if (rtcscene) {
        rtcReleaseScene(rtcscene);
        rtcscene = nullptr;
        for (int i = 0; i < (int)instances.size(); ++i) {
            rtcReleaseGeometry(instances[i].rtgeom_inst);
        }
    }
}

Scene::Scene(Scene &&other)
{
    subscenes = std::move(other.subscenes);
    instances = std::move(other.instances);
    rtcscene = other.rtcscene;
    // Avoid releasing...
    other.rtcscene = nullptr;
}

Scene &Scene::operator=(Scene &&other)
{
    if (this == &other) {
        return *this;
    }

    if (rtcscene) {
        rtcReleaseScene(rtcscene);
        rtcscene = nullptr;
        for (int i = 0; i < (int)instances.size(); ++i) {
            rtcReleaseGeometry(instances[i].rtgeom_inst);
        }
    }

    subscenes = std::move(other.subscenes);
    instances = std::move(other.instances);
    rtcscene = other.rtcscene;
    // Avoid releasing...
    other.rtcscene = nullptr;
    return *this;
}

void Scene::add_subscene(SubScene &&subscene)
{
    subscenes.emplace_back(std::make_unique<SubScene>(std::move(subscene)));
}

void Scene::add_instance(const EmbreeDevice &device, uint32_t subscene_id, const Transform &transform)
{
    ASSERT(subscene_id < subscenes.size());
    RTCGeometry rtcgeom_inst = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_INSTANCE);
    rtcSetGeometryInstancedScene(rtcgeom_inst, subscenes[subscene_id]->rtcscene);
    rtcSetGeometryTransform(rtcgeom_inst, 0, RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR, transform.m.data());
    rtcCommitGeometry(rtcgeom_inst);

    instances.push_back({subscene_id, transform, rtcgeom_inst});
}

void Scene::create_rtc_scene(const EmbreeDevice &device)
{
    rtcscene = rtcNewScene(device);
    // Later useful for different custom precomputation operations.
    rtcSetSceneFlags(rtcscene, RTC_SCENE_FLAG_CONTEXT_FILTER_FUNCTION);

    for (int i = 0; i < (int)subscenes.size(); ++i) {
        SubScene &subscene = *subscenes[i];
        if (!subscene.rtcscene)
            subscene.create_rtc_scene(device);
    }
    for (int j = 0; j < instances.size(); ++j) {
        rtcAttachGeometry(rtcscene, instances[j].rtgeom_inst);
    }

    // build bvh, etc.
    rtcCommitScene(rtcscene);
}

AABB3 Scene::bound() const
{
    ASSERT(rtcscene);
    RTCBounds b;
    rtcGetSceneBounds(rtcscene, &b);
    return AABB3(vec3(b.lower_x, b.lower_y, b.lower_z), vec3(b.upper_x, b.upper_y, b.upper_z));
}

bool Scene::intersect1(const Ray &ray, SceneHit &hit, const IntersectContext &ctx) const
{
    RTCRayHit rayhit = spawn_rtcrayhit(ray.origin, ray.dir, ray.tmin, ray.tmax);
    if (!ks::intersect1(rtcscene, ctx, rayhit)) {
        return false;
    }

    uint32_t inst_id = rayhit.hit.instID[0];
    hit.subscene_id = instances[inst_id].prototype;
    hit.geom_id = rayhit.hit.geomID;

    const SubScene &subscene = *subscenes[hit.subscene_id];
    const Geometry &geom = *subscene.geometries[hit.geom_id];
    const Transform &transform = instances[inst_id].transform;
    hit.it = geom.compute_intersection(rayhit, ray, transform);

    if (!subscene.materials.empty()) {
        hit.material = subscene.materials[rayhit.hit.geomID];
        if (hit.material->normal_map) {
            hit.material->normal_map->apply(hit.it);
        }
    }

    return true;
}

bool Scene::occlude1(const Ray &ray, const IntersectContext &ctx) const
{
    RTCRay rtcray = spawn_ray(ray.origin, ray.dir, ray.tmin, ray.tmax);
    return ks::occlude1(rtcscene, ctx, rtcray);
}

const Transform &Scene::get_instance_transform(uint32_t inst_id) const { return instances[inst_id].transform; }

const MeshGeometry &Scene::get_prototype_mesh_geometry(uint32_t subscene_id, uint32_t geom_id) const
{
    return dynamic_cast<const MeshGeometry &>(*subscenes[subscene_id]->geometries[geom_id]);
}

const Material &Scene::get_prototype_material(uint32_t subscene_id, uint32_t geom_id) const
{
    return *subscenes[subscene_id]->materials[geom_id];
}

bool Scene::are_material_assigned() const
{
    for (int subscene_id = 0; subscene_id < subscenes.size(); ++subscene_id) {
        SubScene &subscene = *subscenes[subscene_id];
        if (subscene.materials.empty() || subscene.materials.size() != subscene.geometries.size() ||
            std::find(subscene.materials.begin(), subscene.materials.end(), nullptr) != subscene.materials.end()) {
            return false;
        }
    }
    return true;
}

bool LocalGeometry::intersect1(const Ray &ray, SceneHit &hit) const
{
    IntersectContext ctx;
    ctx.context.filter = filter_local_geometry;
    ctx.ext = (void *)&geom_id;
    return scene->intersect1(ray, hit, ctx);
}

Scene create_scene(const ConfigArgs &args, const EmbreeDevice &device)
{
    Scene scene;
    if (args.contains("object")) {
        const MeshAsset *mesh_asset = args.asset_table().get<MeshAsset>(args.load_string("object"));
        scene = create_scene_from_mesh_asset(*mesh_asset, device);
    } else if (args.contains("compound_object")) {
        const CompoundMeshAsset *compound_mesh_asset =
            args.asset_table().get<CompoundMeshAsset>(args.load_string("compound_object"));
        scene = create_scene_from_compound_mesh_asset(*compound_mesh_asset, device);
    }
    if (args.contains("material")) {
        assign_material_list(scene, args["material"]);
    }
    ASSERT(scene.are_material_assigned(),
           "No materials. Either load materials from file(s) or provide a material list.");
    return scene;
}

} // namespace ks