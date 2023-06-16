#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void matmul(const float *A, const float *B, float *C, 
            size_t m, size_t n, size_t k) 
{
    /**
     * C = A x B, size of A (m, n), size of B (n, k)
     */
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < k; ++j) {
            float sum = 0;
            for (size_t t = 0; t < n; ++t) {
                sum += A[i * n + t] * B[t * k + j]; // A[i][t] * b[t][j]
            }
            C[i * k + j] = sum; // store C[i][j]
        }
    }
}

void transpose(const float *A, float *A_T, size_t m, size_t n) 
{
    /**
     * transpose A, size of A (m, n)
     */
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            A_T[j * m + i] = A[i * n + j];
        }
    }
}


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */
    /// BEGIN YOUR CODE
    size_t iters = (size_t)(floor(1.0 * m) / batch);
    for (size_t i = 0; i < iters; ++i) {
        size_t cur_bs = batch;
        if (i == iters - 1) {
            cur_bs = m - i * batch;
        }
        auto *p_Xb = &X[i * batch * n]; // pointer to X, of size (cur_bs, n)

        size_t size_Z = cur_bs * k;
        float *Z = new float[size_Z]; // Z of size (cur_bs, k)
        matmul(p_Xb, theta, Z, cur_bs, n, k);

        float *sum_of_cols = new float[cur_bs];
        for (size_t j = 0; j < cur_bs; ++j) {
            float sum = 0;
            for (size_t t = 0; t < k; ++t) {
                Z[j * k + t] = exp(Z[j * k + t]);
                sum += Z[j * k + t];
            }
            sum_of_cols[j] = sum;
        }
        for (size_t j = 0; j < size_Z; ++j) {
            Z[j] /= sum_of_cols[j / k];
        }

        auto *p_yb = &y[i * batch];
        float *I = new float[size_Z]{0};
        for (size_t j = 0; j < cur_bs; ++j) {
            I[j * k + (size_t)p_yb[j]] = 1; // I[j][y[j]] = 1;
        }

        float *Xb_T = new float[cur_bs * n];
        transpose(p_Xb, Xb_T, cur_bs, n);
        for (size_t j = 0; j < size_Z; ++j) {
            Z[j] -= I[j];
        }

        // Xb_T of size (n, cur_bz), Z of size (cur, k)
        float *grad = new float[n * k];
        matmul(Xb_T, Z, grad, n, cur_bs, k);
        for (size_t j = 0; j < n * k; ++j) {
            grad[j] /= cur_bs;
            theta[j] -= lr * grad[j];
        }
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}

int main() {
    const int ROWS = 3, COLS = 4, KOS = 2;
    const float A[ROWS][COLS] = {
        {1, 2, 3, 4},
        {1, 2, 3, 4},
        {1, 2, 3, 4},
    };
    const float B[COLS][KOS] = {
        {1, 1},
        {2, 1},
        {1, 2},
        {1, 0.5}
    };
    float* C = new float[ROWS * KOS];
    matmul((const float*)A, (const float*)B, C, ROWS, COLS, KOS);
    for (int i = 0; i < ROWS; ++i) {
        for (int j = 0; j < KOS; ++j) {
            printf("%f ", C[i * KOS + j]);
        }
        printf("\n");
    }
    delete C;
    float* D = new float[COLS * ROWS];
    transpose((const float*)A, D, ROWS, COLS);
    for (int i = 0; i < COLS; ++i) {
        for (int j = 0; j < ROWS; ++j) {
            printf("%f ", D[i * ROWS + j]);
        }
        printf("\n");
    }
    return 0;
}
