void matmul_native(const float *a, const float *b, float *c, int m,
                              int n, int k);

void matmul_sharedmemory(const float *a, const float *b, float *c, int m,
                              int n, int k);

void matmul_sharedmemory_threadcoarsening(const float *a, const float *b, float *c, int m,
                              int n, int k);
