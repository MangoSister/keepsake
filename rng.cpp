#include "rng.h"

namespace ks
{

float sample_standard_gamma(RNG &rng, float shape)
{
    float b, c;
    float U, V, X, Y;

    if (shape == 1.0) {
        return sample_standard_exp(rng.next());
    } else if (shape == 0.0) {
        return 0.0;
    } else if (shape < 1.0) {
        for (;;) {
            U = rng.next();
            V = sample_standard_exp(rng.next());
            if (U <= 1.0 - shape) {
                X = std::pow(U, 1. / shape);
                if (X <= V) {
                    return X;
                }
            } else {
                Y = -std::log((1 - U) / shape);
                X = std::pow(1.0 - shape + shape * Y, 1. / shape);
                if (X <= (V + Y)) {
                    return X;
                }
            }
        }
    } else {
        b = shape - 1. / 3.;
        c = 1. / std::sqrt(9 * b);
        for (;;) {

            do {
                X = sample_normal(rng.next(), rng.next());
                V = 1.0 + c * X;
            } while (V <= 0.0);

            V = V * V * V;
            U = rng.next();
            if (U < 1.0 - 0.0331 * (X * X) * (X * X))
                return (b * V);
            if (std::log(U) < 0.5 * X * X + b * (1. - V + std::log(V)))
                return (b * V);
        }
    }
}

float sample_beta(RNG &rng, float a, float b)
{
    float Ga, Gb;

    if ((a <= 1.0) && (b <= 1.0)) {
        float U, V, X, Y;
        /* Use Johnk's algorithm */

        while (1) {
            U = rng.next();
            V = rng.next();
            X = std::pow(U, 1.0 / a);
            Y = std::pow(V, 1.0 / b);

            if ((X + Y) <= 1.0) {
                if (X + Y > 0) {
                    return X / (X + Y);
                } else {
                    float logX = std::log(U) / a;
                    float logY = std::log(V) / b;
                    float logM = logX > logY ? logX : logY;
                    logX -= logM;
                    logY -= logM;

                    return std::exp(logX - std::log(std::exp(logX) + std::exp(logY)));
                }
            }
        }
    } else {
        Ga = sample_standard_gamma(rng, a);
        Gb = sample_standard_gamma(rng, b);
        return Ga / (Ga + Gb);
    }
}

} // namespace ks