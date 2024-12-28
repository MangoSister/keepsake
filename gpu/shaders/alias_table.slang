implementing ks;

public struct AliasTableSample
{
    int index;
    float pmf;
    float u_remap;
}

public struct AliasTable
{
    public AliasTableSample sample(float u)
    {
        // Compute alias table _offset_ and remapped random sample _up_
        uint N, stride;
        bins.GetDimensions(N, stride);
        unused(stride);

        int offset = min((int)floor(u * N), N - 1);
        float up = min(u * N - offset, fp32_before_one);

        AliasTableSample sample;
        if (up < bins[offset].q) {
            // Return sample for alias table at _offset_
            // ASSERT(bins[offset].p > 0);
            sample.pmf = bins[offset].p;
            sample.u_remap = min(up / bins[offset].q, fp32_before_one);
            sample.index = offset;
            return sample;

        } else {
            // Return sample for alias table at _alias[offset]_
            int alias = bins[offset].alias;
            // ASSERT(alias >= 0);
            // ASSERT(bins[alias].p > 0);
            sample.pmf = bins[alias].p;
            sample.u_remap = min((up - bins[offset].q) / (1 - bins[offset].q), fp32_before_one);
            sample.index = alias;
            return sample;
        }
    }

    public float pmf(int index) { return bins[index].p; }

    struct Bin
    {
        float q, p;
        int alias;
    };
    StructuredBuffer<Bin, ScalarDataLayout> bins;
};
