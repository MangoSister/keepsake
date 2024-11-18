implementing ks;

public struct PerspectiveCamera
{
    public RayDesc spawn_ray(float2 film_pos, float tmin, float tmax)
    {
        float2 d = film_pos * 2.0 - 1.0;
        float4 target = mul(proj_to_camera, float4(d.x, d.y, 1, 1));

        RayDesc ray;
        ray.Origin = mul(camera_to_world, float4(0, 0, 0, 1)).xyz;
        ray.Direction = normalize(mul((float3x3)camera_to_world, target.xyz));
        ray.TMin = tmin;
        ray.TMax = tmax;
        return ray;
    }

    public float4x4 world_to_proj;
    public float4x4 camera_to_world;
    public float4x4 proj_to_camera;
};