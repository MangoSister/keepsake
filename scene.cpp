#include "scene.h"
#include "mesh_asset.h"
#include "normal_map.h"

KS_NAMESPACE_BEGIN

Scene::~Scene()
{
    if (rtcscene) {
        rtcReleaseScene(rtcscene);
        rtcscene = nullptr;
    }
}

Scene::Scene(Scene &&other)
{
    geometries = std::move(other.geometries);
    materials = std::move(other.materials);
    rtcscene = other.rtcscene;
    // Avoid releasing...
    other.rtcscene = nullptr;
}

void Scene::create_rtc_scene(const EmbreeDevice &device)
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
    if (!::intersect1(rtcscene, ctx, rayhit)) {
        return false;
    }

    hit.it = geometries[rayhit.hit.geomID]->compute_intersection(rayhit);
    hit.it.compute_uv_partials(ray);
    hit.material = materials[rayhit.hit.geomID];
    hit.geom_id = rayhit.hit.geomID;

    if (hit.material->normal_map) {
        hit.material->normal_map->apply(hit.it);
    }

    return true;
}

bool Scene::occlude1(const Ray &ray, const IntersectContext &ctx) const
{
    RTCRay rtcray = spawn_ray(ray.origin, ray.dir, ray.tmin, ray.tmax);
    return ::occlude1(rtcscene, ctx, rtcray);
}


bool LocalGeometry::intersect1(const Ray &ray, SceneHit &hit) const
{
    IntersectContext ctx;
    ctx.context.filter = filter_local_geometry;
    ctx.ext = (void *)&geom_id;
    return scene->intersect1(ray, hit, ctx);
}

KS_NAMESPACE_END