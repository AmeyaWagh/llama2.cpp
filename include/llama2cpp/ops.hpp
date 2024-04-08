#ifndef LLAMA2CPP_OPS_HPP
#define LLAMA2CPP_OPS_HPP
#include <string>
#include <math.h>
#include "llama2cpp/tensor.hpp"

namespace llama2cpp
{

    void rmsnorm(float *out, float *x, float *weight, int size)
    {
        // calculate sum of squares
        float ss = 0.0f;
        for (int j = 0; j < size; j++)
        {
            ss += x[j] * x[j];
        }
        ss /= size;
        ss += 1e-5f;
        ss = 1.0f / sqrtf(ss);
        // normalize and scale
        for (int j = 0; j < size; j++)
        {
            out[j] = weight[j] * (ss * x[j]);
        }
    }

    /**
     * @brief Softmax in-place
     *
     * @param x input tensor.
     * @param size size of the tensor.
     */
    void softmax(float *x, int size)
    {
        // find max value (for numerical stability)
        float max_val = x[0];
        for (int i = 1; i < size; i++)
        {
            if (x[i] > max_val)
            {
                max_val = x[i];
            }
        }

        // exp and sum
        float sum = 0.0f;
        for (int i = 0; i < size; i++)
        {
            x[i] = expf(x[i] - max_val);
            sum += x[i];
        }

        // normalize
        for (int i = 0; i < size; i++)
        {
            x[i] /= sum;
        }
    }

    /**
     * @brief Matrix multiplication operation
     *
     * W (d,n) @ x (n,) -> xout (d,)
     * by far the most amount of time is spent inside this little function
     *
     * TODO: implement a generic method for tensors.
     *
     * @param xout output tensor.
     * @param x input tensor.
     * @param w weight matrix.
     * @param n input vector dimension.
     * @param d output vector dimension.
     */
    void matmul(float *xout, const float *x, const float *w, int n, int d)
    {

        int i;
#pragma omp parallel for private(i)
        for (i = 0; i < d; i++)
        {
            float val = 0.0f;
            for (int j = 0; j < n; j++)
            {
                val += w[i * n + j] * x[j];
            }
            xout[i] = val;
        }
    }

    /**
     * @brief Matrix multiplication
     * xout = w * xin
     *
     * @tparam T datatype
     * @param xout output tensor (m)
     * @param xin input tensor (n)
     * @param w weight matrix (mxn)
     */
    template <typename T>
    void matmul(TensorView<T> &xout, const TensorView<T> &xin, const TensorView<T> &w)
    {
        // TODO compute based matmul
        matmul(xout.data(), xin.data(), w.data(), xin.size(), xout.size());
    }

    template <typename T>
    void rmsnorm(TensorView<T> &out, TensorView<T> &x, const TensorView<T> &weight)
    {

        // calculate sum of squares
        T ss = static_cast<T>(0);
        for (auto j = 0; j < x.size(); ++j)
        {
            ss += x(j) * x(j);
        }
        ss /= x.size();
        ss += 1e-5f;
        ss = 1.0f / sqrtf(ss);
        // normalize and scale
        for (auto j = 0; j < x.size(); ++j)
        {
            out(j) = weight(j) * (ss * x(j));
        }
    }

}
#endif