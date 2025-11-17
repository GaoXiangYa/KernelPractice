// y=αAx+βy
// A[M, N]
// X[N, 1]
// y[M, 1]
void gemv_v0(float *mat, float *vec_x, float *vec_y, const int m, const int n,
             const float alpha, const float beta);
