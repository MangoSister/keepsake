implementing ks;

// Implement the Adam optimizer.
// https://pytorch.org/docs/stable/generated/torch.optim.Adam.html

struct AdamOptimizerHyperParams
{
    __init()
    {
        lr = 1e-3;
        beta1 = 0.9;
        beta2 = 0.999;
        eps = 1e-8;
        iter = 1;
    }

    float lr;
    float beta1, beta2;
    float eps;
    int iter; // NOTE: iter starts from 1.
}
// https://shader-slang.com/slang/user-guide/spirv-target-specific.html#parameterblock-for-spir-v-target
// https://shader-slang.com/slang/user-guide/spirv-target-specific.html#push-constants

[shader("compute")]
[numthreads(1024, 1, 1)]
void adam_optimizer_step(uint3 thread_id: SV_DispatchThreadID,
                         uniform AdamOptimizerHyperParams hparams, // This becomes a push constant
                         uniform RWStructuredBuffer<float> parameters, uniform StructuredBuffer<float> gradients,
                         uniform RWStructuredBuffer<float> first_moments,
                         uniform RWStructuredBuffer<float> second_moments)
{
    int N = 0, parameters_stride = 0;
    parameters.GetDimensions(N, parameters_stride);
    uint idx = thread_id.x;
    if (idx >= N) {
        return;
    }
    float g = gradients[idx];
    float m = first_moments[idx];
    float v = second_moments[idx];

    float mt = lerp(g, m, hparams.beta1);
    float vt = lerp(g * g, v, hparams.beta2);
    float mt_hat = mt / (1 - pow(hparams.beta1, (float)hparams.iter));
    float vt_hat = vt / (1 - pow(hparams.beta2, (float)hparams.iter));

    parameters[idx] -= hparams.lr * mt_hat / (sqrt(vt_hat) + hparams.eps);
    first_moments[idx] = mt;
    second_moments[idx] = vt;
}
