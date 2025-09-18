void rmsnorm_v0(float *input, float *output, const int input_len,
                const float eps);

void rmsnorm_v1(float *input, float *output, const int input_len,
                const float eps);

void flashinfer_rmsnorm(float *input, float *weight, float *output,
                        const int input_len, const float eps);

void rmsnorm_v0_benchmark();

void rmsnorm_v1_benchmark();

void flashinfer_rmsnorm_benchmark();
