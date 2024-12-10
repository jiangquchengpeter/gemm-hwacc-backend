#ifndef GEMM_TB_H
#define GEMM_TB_H
#define DEV_BOUND(BASE, LEN) \
    {                        \
        BASE, BASE + LEN     \
    }

// Prev inc from DARKNET_DIMS.HH 
#include <map>
#include <vector>
#include <string>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <random>
#include <string.h>

// Macros that define testbench generation parameters
#define SEED 2345467345
#define MAT_VAL_LOWER_LIM 0
#define MAT_VAL_UPPER_LIM (1 << 8) - 1
#define SMALL_MAT_DIM_LOWER_LIM 1
#define SMALL_MAT_DIM_UPPER_LIM 200
#define LARGE_MAT_DIM_LOWER_LIM 200
#define LARGE_MAT_DIM_UPPER_LIM 512

// Status bar macros
#define PBSTR "||||||||||"
#define PBWIDTH 10

using namespace std::placeholders;

using DimGenerator = void(uint32_t &, uint32_t &, uint32_t &, uint32_t);



/** @brief Gemm_nn extracted from Darknet but in 32bit fixed point.
 *  completes the C = ALPHA * A * B + C matrix operation, and the output C is
 *  also stored in rows (all rows are combined into one row)
 * @param MA, the number of lines in C (not transposed)
 * @param NB, the number of columns in C (not as a device)
 * @param KA's column number, C's row number (not transposed)
 * @param ALPHA coefficient
 * @param A input matrix (one-dimensional array format)
 * @param lda A number of columns (not transposed)
 * @param B input matrix (one-dimensional array format)
 * @param ldb B's number of columns (not transposed)
 * @param C input matrix (one-dimensional array format)
 * @param ldc C column number (not transposed)
 */
template <typename ELEMTYPE>
void sw_gemm_nn(uint32_t M, uint32_t N, uint32_t K, uint32_t ALPHA,
                void *A, uint32_t lda,
                void *B, uint32_t ldb,
                void *C, uint32_t ldc)
{
    ELEMTYPE* pA = (ELEMTYPE*)A;
    ELEMTYPE* pB = (ELEMTYPE*)B;
    ELEMTYPE* pC = (ELEMTYPE*)C;
    uint32_t i, j, k;
    for (i = 0; i < M; ++i)
    {
        for (k = 0; k < K; ++k)
        {
            ELEMTYPE A_PART = ALPHA * pA[i * lda + k];
            for (j = 0; j < N; ++j)
            {
                pC[i * ldc + j] += A_PART * pB[k * ldb + j];
            }
        }
    }
}

#endif // GEMM_TB_H