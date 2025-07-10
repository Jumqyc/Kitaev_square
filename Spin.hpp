#ifndef _SPIN_H
#define _SPIN_H
#include <cstdlib>
#include <array>
#include <vector>
#include <tuple>
#include <cmath>
#include <random>
#include <thread>
#include <stdexcept>
#include <omp.h>

namespace vec // numpy like operations
{
    inline float dot(const std::array<float, 3> &a, const std::array<float, 3> &b)
    {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    }
}

class Spin
{
public:
    // constructor
    Spin(int l) : L(l)
    {
        s.reserve(L * L);
        for (int i = 0; i < L * L; ++i)
        {
            s.push_back({0, 0, 1.0f}); // initialize all spins to (0, 0, 1)
        }
    }

    void set_parameters(float set_temp, float set_J11, float set_J12,
                        float set_J21, float set_J22, float set_K)
    {
        t = set_temp;
        J11 = set_J11;
        J12 = set_J12;
        J21 = set_J21;
        J22 = set_J22;
        K = set_K;
    }

    std::tuple<float, float, float, float, float, float> get_parameters() const
    {
        return std::make_tuple(t, J11, J12, J21, J22, K);
    }

    void set_record(const std::vector<std::vector<std::array<float, 3>>> &new_record)
    {
        for (auto rec : new_record)
        {
            saving.push_back(rec);
        }
        set_spin();
    }

    void run(int step, int spacing)
    {
        if (s.size() != L * L)
        {
            throw std::runtime_error("Spin object not initialized with correct size.");
        }
        saving.reserve(saving.size() + step);
        for (int i = 0; i < step; ++i)
        {
            update(spacing);
            saving.push_back(s);
        }
    }

    std::vector<std::vector<std::array<float, 3>>> get_saving() const
    {
        return saving;
    }

private:
    const int L;                         // lattice size, LxL
    std::vector<std::array<float, 3>> s; // spin vectors
    float t, J11, J12, J21, J22, K;      // temperature and coupling constants
    std::vector<std::vector<std::array<float, 3>>> saving;

    void set_spin(void)
    {
        s = {}; // clear current spins
        s.reserve(L * L);
        if (saving.empty())
        {
            for (int i = 0; i < L * L; ++i)
            {
                s.push_back({{0, 0, 1.0f}}); // initialize all spins to (0, 0, 1)
            }
            return;
        }
        for (auto vec : saving.back())
        {
            s.push_back(vec);
        }
        return;
    }

    void update(int spacing)
    {
        for (int i = 0; i < spacing; ++i)
        {
            for (int dx = 0; dx < 3; ++dx)
            {
                for (int dy = 0; dy < 3; ++dy)
                {
                    checkboard(dx, dy);
                    // local updates are handled sequentially in checkboard; multithreading is implemented inside checkboard
                }
            }
        }
    }

    void checkboard(int dx, int dy)
    {
        #pragma omp parallel for collapse(2) // parallelize the outer loops
        for (int x = dx; x < L; x += 3)
        {
            for (int y = dy; y < L; y += 3)
            {
                local_update(x, y, (x + 1) % L, (x - 1 + L) % L, (y + 1) % L, (y - 1 + L) % L);
            }
        }
    }

    void local_update(const int x, const int y,
                      const int xu, const int xd,
                      const int yu, const int yd)
    {
        // Thread-local random number generator and uniform distribution
        thread_local static std::mt19937 rng{std::random_device{}()};
        thread_local static std::uniform_real_distribution<float> dist{0.0f, 1.0f};
        // store original spin

        const int idx = ind(x, y);
        const std::array<float, 3> old_spin = s[idx];

        // new spin
        const float phi = 6.28318530718f * dist(rng);
        const float theta = std::acos(2 * dist(rng) - 1);
        const std::array<float, 3> new_spin = {
            std::sin(theta) * std::cos(phi),
            std::sin(theta) * std::sin(phi),
            std::cos(theta)};

        // calculate energy change
        float dE = 0;
        float old_dot, new_dot, diff;
        std::array<float, 3> nn_spin;

        // nn interaction. Heisenberg, biquadratic and Kitaev terms
        // (x-1,y)
        nn_spin = s[ind(xd, y)];
        old_dot = vec::dot(old_spin, nn_spin);
        new_dot = vec::dot(new_spin, nn_spin);
        diff = new_dot - old_dot;
        dE += J11 * diff;                                   // nn Heisenberg S_i dot S_j
        dE += J12 * diff * (new_dot + old_dot);             // nn biquadratic (S_i dot S_j)^2
        dE += K * (new_spin[0] - old_spin[0]) * nn_spin[0]; // Kitaev term Kx S_{i+e_x}x ^ S_i^x
        // (x+1,y)
        nn_spin = s[ind(xu, y)];
        old_dot = vec::dot(old_spin, nn_spin);
        new_dot = vec::dot(new_spin, nn_spin);
        diff = new_dot - old_dot;
        dE += J11 * diff;                                   // nn Heisenberg S_i dot S_j
        dE += J12 * diff * (new_dot + old_dot);             // nn biquadratic (S_i dot S_j)^2
        dE += K * (new_spin[0] - old_spin[0]) * nn_spin[0]; // Kitaev term Kx S_{i-e_x}^x  S_i^x
        // (x,y-1)
        nn_spin = s[ind(x, yd)];
        old_dot = vec::dot(old_spin, nn_spin);
        new_dot = vec::dot(new_spin, nn_spin);
        diff = new_dot - old_dot;
        dE += J11 * diff;                                   // nn Heisenberg S_i dot S_j
        dE += J12 * diff * (new_dot + old_dot);             // nn biquadratic (S_i dot S_j)^2
        dE += K * (new_spin[1] - old_spin[1]) * nn_spin[1]; // Kitaev term Ky S_{i+y}^y  S_i^y
        // (x,y+1)
        nn_spin = s[ind(x, yu)];
        old_dot = vec::dot(old_spin, nn_spin);
        new_dot = vec::dot(new_spin, nn_spin);
        diff = new_dot - old_dot;
        dE += J11 * diff;                                   // nn Heisenberg S_i dot S_j
        dE += J12 * diff * (new_dot + old_dot);             // nn biquadratic (S_i dot S_j)^2
        dE += K * (new_spin[1] - old_spin[1]) * nn_spin[1]; // Kitaev term Ky S_{i+y}^y  S_i^y

        // nnn interaction, Heisenberg, biquadratic, without Kitaev term
        // (x-1,y-1)
        nn_spin = s[ind(xd, yd)];
        old_dot = vec::dot(old_spin, nn_spin);
        new_dot = vec::dot(new_spin, nn_spin);
        diff = new_dot - old_dot;
        dE += J21 * diff;                       // nnn Heisenberg S_i dot S_j
        dE += J22 * diff * (new_dot + old_dot); // nnn biquadratic (S_i dot S_j)^2
        // (x+1,y+1)
        nn_spin = s[ind(xu, yu)];
        old_dot = vec::dot(old_spin, nn_spin);
        new_dot = vec::dot(new_spin, nn_spin);
        diff = new_dot - old_dot;
        dE += J21 * diff;                       // nnn Heisenberg S_i dot S_j
        dE += J22 * diff * (new_dot + old_dot); // nnn biquadratic (S_i dot S_j)^2
        // (x-1,y+1)
        nn_spin = s[ind(xd, yu)];
        old_dot = vec::dot(old_spin, nn_spin);
        new_dot = vec::dot(new_spin, nn_spin);
        diff = new_dot - old_dot;
        dE += J21 * diff;                       // nnn Heisenberg S_i dot S_j
        dE += J22 * diff * (new_dot + old_dot); // nnn biquadratic (S_i dot S_j)^2
        // (x+1,y-1)
        nn_spin = s[ind(xu, yd)];
        old_dot = vec::dot(old_spin, nn_spin);
        new_dot = vec::dot(new_spin, nn_spin);
        diff = new_dot - old_dot;
        dE += J21 * diff;                       // nnn Heisenberg S_i dot S_j
        dE += J22 * diff * (new_dot + old_dot); // nnn biquadratic (S_i dot S_j)^2

        // Metropolis
        if (dist(rng) < std::exp(-dE / t))
        {
            s[idx] = new_spin;
        }
    }

    inline int ind(const int x, const int y) const
    {
        return x + y * L;
    }
};
#endif