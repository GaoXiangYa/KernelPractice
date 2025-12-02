#include <fstream>
// y=αAx+βy
// A[M, N]
// X[N, 1]
// y[M, 1]
void gemv_v0(float *mat_a, float *vec_x, float *vec_y, const int m, const int n,
             const float alpha, const float beta);

void gemv_v1(float *mat_a, float *vec_x, float *vec_y, const int m, const int n,
             const float alpha, const float beta);

void gemv_v2(float *mat_a, float *vec_x, float *vec_y, const int m, const int n,
             const float alpha, const float beta);

void gemv_v3(float *mat_a, float *vec_x, float *vec_y, const int m, const int n,
             const float alpha, const float beta);

void gemv_v4(float *mat_a, float *vec_x, float *vec_y, const int m, const int n,
             const float alpha, const float beta);

void gemv_v5(float *mat_a, float *vec_x, float *vec_y, const int m, const int n,
             const float alpha, const float beta);

void gemv_v6(float *mat_a, float *vec_x, float *vec_y, const int m, const int n,
             const float alpha, const float beta);

void cutlass_gemv_fp32(float *mat_a, float *vec_x, float *vec_y, const int m,
                       const int n, const float alpha, const float beta);

void benchmark_gemv_v0(std::ofstream &file, const int m, const int n);

void benchmark_gemv_v1(std::ofstream &file, const int m, const int n);

void benchmark_gemv_v2(std::ofstream &file, const int m, const int n);

void benchmark_gemv_v3(std::ofstream &file, const int m, const int n);

void benchmark_gemv_v4(std::ofstream &file, const int m, const int n);

void benchmark_gemv_v5(std::ofstream &file, const int m, const int n);

void benchmark_gemv_v6(std::ofstream &file, const int m, const int n);

void benchmark_cutlass_gemv_fp32(std::ofstream &file, const int m, const int n);
