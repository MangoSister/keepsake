implementing ks;

public interface IPixelFilter
{
    float eval(float2 offset);
    float2 sample(float2 u);
    float radius();
}

public struct BoxPixelFilter : IPixelFilter
{
    public __init(float width = 1.0f) { r = 0.5 * width; }

    public float eval(float2 offset)
    {
        if (offset.x < -r || offset.x > r || offset.y < -r || offset.y > r) {
            return 0.0f;
        }
        return 1.0f;
    }

    public float2 sample(float2 u) { return lerp(float2(-r), float2(r), u); }

    public float radius() { return r; }

    float r;
}