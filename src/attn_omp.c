/*
 * attn_omp.c -- Q4: your best variant, parallelised with OpenMP.
 *
 * Same kernels as attn_best.c. Parallelism is over blocks of query rows:
 * every block is independent (its own scores, softmax and output rows), so
 * there is no reduction and no synchronisation inside the main loop. Each
 * thread allocates its own qb / S / ob scratch once, so the score blocks stay
 * in that core's private L2. schedule(dynamic) because the P-cores and
 * E-cores of this CPU run at very different speeds; BR is 24 here (not 48)
 * so that N = 1024 still gives 43 blocks for up to 22 threads.
 * K / V transposition + padding is also split across threads.
 * Without AVX2 + FMA (e.g. an ARM machine) a plain C version of the same
 * kernels is compiled instead: same algorithm and results, just slower.
 */
#define _POSIX_C_SOURCE 200809L
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define USE_AVX2 1
#define amalloc(n) _mm_malloc((n), 64)
#define afree(p) _mm_free(p)
#else
#define USE_AVX2 0
#define amalloc(n) malloc(n)
#define afree(p) free(p)
#endif
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* ------------------------------------------------------------------------- */
/* attention(): write your implementation here.                               */
/* ------------------------------------------------------------------------- */
#define BR 24  /* query rows per block (multiple of MR) */
#define MR 6   /* micro-kernel rows                     */
#define JB 256 /* j block: Kt / V panel stays in L1/L2  */

static inline int round_up(int x, int m) { return (x + m - 1) / m * m; }

#if USE_AVX2
/* exp(x) for 8 floats (Cephes expf). */
static inline __m256 exp256(__m256 x)
{
    x = _mm256_min_ps(x, _mm256_set1_ps(88.3762626647949f));
    x = _mm256_max_ps(x, _mm256_set1_ps(-87.3365447505531f));
    __m256 fx = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f)),
                                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(0.693359375f), x);
    x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(-2.12194440e-4f), x);
    __m256 y = _mm256_set1_ps(1.9875691500e-4f);
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.3981999507e-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.3334519073e-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.1665795894e-2f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.6666665459e-1f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.0000001201e-1f));
    y = _mm256_fmadd_ps(y, _mm256_mul_ps(x, x), _mm256_add_ps(x, _mm256_set1_ps(1.0f)));
    __m256i e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(fx), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(y, _mm256_castsi256_ps(e));
}

static inline float hsum256(__m256 v)
{
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    return _mm_cvtss_f32(_mm_add_ss(s, _mm_movehdup_ps(s)));
}

static inline float hmax256(__m256 v)
{
    __m128 s = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_max_ps(s, _mm_movehl_ps(s, s));
    return _mm_cvtss_f32(_mm_max_ss(s, _mm_movehdup_ps(s)));
}

/*
 * C[r][0..16) (+)= sum_{p < len} A[r * lda + p] * B[p * ldb + 0..16),  r < 6
 * Q K^T: A = scaled Q block (lda = dp), B = Kt panel (ldb = Np), len = dp
 * P V  : A = score block    (lda = Np), B = V panel  (ldb = dp), len = JB
 */
static inline void kernel6x16(const float *restrict A, int lda, const float *restrict B, int ldb,
                              int len, float *restrict C, int ldc, int accumulate)
{
    __m256 c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51;
    if (accumulate) {
        c00 = _mm256_loadu_ps(C + 0 * ldc); c01 = _mm256_loadu_ps(C + 0 * ldc + 8);
        c10 = _mm256_loadu_ps(C + 1 * ldc); c11 = _mm256_loadu_ps(C + 1 * ldc + 8);
        c20 = _mm256_loadu_ps(C + 2 * ldc); c21 = _mm256_loadu_ps(C + 2 * ldc + 8);
        c30 = _mm256_loadu_ps(C + 3 * ldc); c31 = _mm256_loadu_ps(C + 3 * ldc + 8);
        c40 = _mm256_loadu_ps(C + 4 * ldc); c41 = _mm256_loadu_ps(C + 4 * ldc + 8);
        c50 = _mm256_loadu_ps(C + 5 * ldc); c51 = _mm256_loadu_ps(C + 5 * ldc + 8);
    } else {
        c00 = c01 = c10 = c11 = c20 = c21 = c30 = c31 = c40 = c41 = c50 = c51 = _mm256_setzero_ps();
    }
    for (int p = 0; p < len; p++) {
        const __m256 b0 = _mm256_loadu_ps(B + (size_t)p * ldb);
        const __m256 b1 = _mm256_loadu_ps(B + (size_t)p * ldb + 8);
        __m256 a;
        a = _mm256_broadcast_ss(A + 0 * lda + p); c00 = _mm256_fmadd_ps(a, b0, c00); c01 = _mm256_fmadd_ps(a, b1, c01);
        a = _mm256_broadcast_ss(A + 1 * lda + p); c10 = _mm256_fmadd_ps(a, b0, c10); c11 = _mm256_fmadd_ps(a, b1, c11);
        a = _mm256_broadcast_ss(A + 2 * lda + p); c20 = _mm256_fmadd_ps(a, b0, c20); c21 = _mm256_fmadd_ps(a, b1, c21);
        a = _mm256_broadcast_ss(A + 3 * lda + p); c30 = _mm256_fmadd_ps(a, b0, c30); c31 = _mm256_fmadd_ps(a, b1, c31);
        a = _mm256_broadcast_ss(A + 4 * lda + p); c40 = _mm256_fmadd_ps(a, b0, c40); c41 = _mm256_fmadd_ps(a, b1, c41);
        a = _mm256_broadcast_ss(A + 5 * lda + p); c50 = _mm256_fmadd_ps(a, b0, c50); c51 = _mm256_fmadd_ps(a, b1, c51);
    }
    _mm256_storeu_ps(C + 0 * ldc, c00); _mm256_storeu_ps(C + 0 * ldc + 8, c01);
    _mm256_storeu_ps(C + 1 * ldc, c10); _mm256_storeu_ps(C + 1 * ldc + 8, c11);
    _mm256_storeu_ps(C + 2 * ldc, c20); _mm256_storeu_ps(C + 2 * ldc + 8, c21);
    _mm256_storeu_ps(C + 3 * ldc, c30); _mm256_storeu_ps(C + 3 * ldc + 8, c31);
    _mm256_storeu_ps(C + 4 * ldc, c40); _mm256_storeu_ps(C + 4 * ldc + 8, c41);
    _mm256_storeu_ps(C + 5 * ldc, c50); _mm256_storeu_ps(C + 5 * ldc + 8, c51);
}

/* row[j] = exp(row[j] - max(row)); returns the sum. Np is a multiple of 16. */
static inline float softmax_row(float *restrict row, int Np)
{
    __m256 m = _mm256_set1_ps(-INFINITY);
    for (int j = 0; j < Np; j += 8)
        m = _mm256_max_ps(m, _mm256_loadu_ps(row + j));
    const __m256 mx = _mm256_set1_ps(hmax256(m));
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    for (int j = 0; j < Np; j += 16) {
        const __m256 e0 = exp256(_mm256_sub_ps(_mm256_loadu_ps(row + j), mx));
        const __m256 e1 = exp256(_mm256_sub_ps(_mm256_loadu_ps(row + j + 8), mx));
        _mm256_storeu_ps(row + j, e0);
        _mm256_storeu_ps(row + j + 8, e1);
        s0 = _mm256_add_ps(s0, e0);
        s1 = _mm256_add_ps(s1, e1);
    }
    return hsum256(_mm256_add_ps(s0, s1));
}

#else /* plain C fallback, same structure */

static inline float softmax_row(float *restrict row, int Np)
{
    float mx = row[0];
    for (int j = 1; j < Np; j++)
        mx = row[j] > mx ? row[j] : mx;
    float sum = 0.0f;
    for (int j = 0; j < Np; j++) {
        row[j] = expf(row[j] - mx);
        sum += row[j];
    }
    return sum;
}

static inline void kernel6x16(const float *restrict A, int lda, const float *restrict B, int ldb,
                              int len, float *restrict C, int ldc, int accumulate)
{
    float acc[MR][16];
    for (int r = 0; r < MR; r++)
        for (int c = 0; c < 16; c++)
            acc[r][c] = accumulate ? C[r * ldc + c] : 0.0f;
    for (int p = 0; p < len; p++)
        for (int r = 0; r < MR; r++) {
            const float a = A[r * lda + p];
            for (int c = 0; c < 16; c++)
                acc[r][c] += a * B[(size_t)p * ldb + c];
        }
    for (int r = 0; r < MR; r++)
        for (int c = 0; c < 16; c++)
            C[r * ldc + c] = acc[r][c];
}
#endif

/*
 * One block of rows [i0, i0 + rows). qb: BR x dp, S: BR x Np, ob: BR x dp.
 * Kt: dp x Np (K transposed and padded), Vp: Np x dp (V padded).
 */
static void attention_block(const float *Q, const float *Kt, const float *Vp, float *O, int N, int d,
                            int Np, int dp, float scale, int i0, int rows,
                            float *qb, float *S, float *ob)
{
    const int rp = round_up(rows, MR);

    memset(qb, 0, (size_t)rp * dp * sizeof(float));
    for (int r = 0; r < rows; r++)
        for (int k = 0; k < d; k++)
            qb[(size_t)r * dp + k] = Q[(size_t)(i0 + r) * d + k] * scale;

    /* scores: S = qb Kt, j blocked so the Kt panel (dp x JB) stays in L2 */
    for (int jb = 0; jb < Np; jb += JB) {
        const int jend = jb + JB < Np ? jb + JB : Np;
        for (int r0 = 0; r0 < rp; r0 += MR)
            for (int j = jb; j < jend; j += 16)
                kernel6x16(qb + (size_t)r0 * dp, dp, Kt + j, Np, dp, S + (size_t)r0 * Np + j, Np, 0);
    }

    /* softmax numerators; padded columns get -inf so they contribute 0 */
    float inv[BR];
    for (int r = 0; r < rp; r++) {
        float *row = S + (size_t)r * Np;
        for (int j = N; j < Np; j++)
            row[j] = -INFINITY;
        inv[r] = 1.0f / softmax_row(row, Np);
    }

    /* ob = P Vp, j blocked so the Vp panel (JB x dp) stays close */
    for (int jb = 0; jb < Np; jb += JB) {
        const int jlen = (jb + JB < Np ? jb + JB : Np) - jb;
        for (int r0 = 0; r0 < rp; r0 += MR)
            for (int c = 0; c < dp; c += 16)
                kernel6x16(S + (size_t)r0 * Np + jb, Np, Vp + (size_t)jb * dp + c, dp, jlen,
                           ob + (size_t)r0 * dp + c, dp, jb > 0);
    }

    for (int r = 0; r < rows; r++)
        for (int k = 0; k < d; k++)
            O[(size_t)(i0 + r) * d + k] = ob[(size_t)r * dp + k] * inv[r];
}

void attention(const float *Q, const float *K, const float *V, float *O, int N, int d)
{
    const float scale = 1.0f / sqrtf((float)d);
    const int Np = round_up(N, 16), dp = round_up(d, 16);
    float *Kt = amalloc((size_t)dp * Np * sizeof(float));
    float *Vp = amalloc((size_t)Np * dp * sizeof(float));

#pragma omp parallel
    {
        /* zero padding (only touched when N or d is not a multiple of 16) */
#pragma omp for schedule(static)
        for (int k = 0; k < dp; k++) {
            if (k >= d)
                memset(Kt + (size_t)k * Np, 0, (size_t)Np * sizeof(float));
            else
                for (int j = N; j < Np; j++)
                    Kt[(size_t)k * Np + j] = 0.0f;
        }
#pragma omp for schedule(static)
        for (int j = 0; j < Np; j++)
            for (int k = (j < N ? d : 0); k < dp; k++)
                Vp[(size_t)j * dp + k] = 0.0f;

        /* transpose K in 64-column strips (each thread writes its own columns) */
#pragma omp for schedule(static)
        for (int j0 = 0; j0 < N; j0 += 64) {
            const int j1 = j0 + 64 < N ? j0 + 64 : N;
            for (int j = j0; j < j1; j++)
                for (int k = 0; k < d; k++) {
                    Kt[(size_t)k * Np + j] = K[(size_t)j * d + k];
                    Vp[(size_t)j * dp + k] = V[(size_t)j * d + k];
                }
        }

        float *qb = amalloc((size_t)BR * dp * sizeof(float));
        float *S = amalloc((size_t)BR * Np * sizeof(float));
        float *ob = amalloc((size_t)BR * dp * sizeof(float));

#pragma omp for schedule(dynamic, 1)
        for (int i0 = 0; i0 < N; i0 += BR)
            attention_block(Q, Kt, Vp, O, N, d, Np, dp, scale, i0, N - i0 < BR ? N - i0 : BR, qb, S, ob);

        afree(ob);
        afree(S);
        afree(qb);
    }

    afree(Vp);
    afree(Kt);
}

/* ------------------------------------------------------------------------- */
/* Driver: do not change below this line.                                     */
/* ------------------------------------------------------------------------- */
#define REPS 5

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s <input.bin> <output.bin>\n", argv[0]);
        return 1;
    }
    FILE *in = fopen(argv[1], "rb");
    if (!in) {
        perror(argv[1]);
        return 1;
    }
    int32_t hdr[2];
    if (fread(hdr, sizeof hdr, 1, in) != 1) {
        fprintf(stderr, "%s: bad header\n", argv[1]);
        return 1;
    }
    int N = hdr[0], d = hdr[1];
    size_t n = (size_t)N * d;
    float *Q = malloc(n * sizeof(float)), *K = malloc(n * sizeof(float));
    float *V = malloc(n * sizeof(float)), *O = malloc(n * sizeof(float));
    if (!Q || !K || !V || !O || fread(Q, sizeof(float), n, in) != n ||
        fread(K, sizeof(float), n, in) != n || fread(V, sizeof(float), n, in) != n) {
        fprintf(stderr, "%s: could not read Q, K, V\n", argv[1]);
        return 1;
    }
    fclose(in);

    attention(Q, K, V, O, N, d); /* warm-up, not timed */
    double t[REPS];
    for (int r = 0; r < REPS; r++) {
        double t0 = now();
        attention(Q, K, V, O, N, d);
        t[r] = now() - t0;
    }
    qsort(t, REPS, sizeof t[0], cmp_double);
    double median = t[REPS / 2];

    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    const char *name = strrchr(argv[0], '/') ? strrchr(argv[0], '/') + 1 : argv[0];
    printf("%-18s N=%-5d d=%-4d threads=%-3d time_ms=%10.3f  GFLOP/s=%8.2f\n", name, N, d,
           threads, median * 1e3, 4.0 * N * (double)N * d / median * 1e-9);

    FILE *out = fopen(argv[2], "wb");
    if (!out || fwrite(hdr, sizeof hdr, 1, out) != 1 || fwrite(O, sizeof(float), n, out) != n) {
        perror(argv[2]);
        return 1;
    }
    fclose(out);
    free(Q);
    free(K);
    free(V);
    free(O);
    return 0;
}
