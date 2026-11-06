/*
 * attn_tiled.c -- Q3b: cache and register tiling.
 *
 * Built on attn_loops.c. Both matrix products are computed in MR x NR blocks
 * of the output held in a small local array (acc) that clang keeps in ymm
 * registers, so each loaded element of Q/Kt (or S/V) is reused MR or NR
 * times instead of once. The outer j loop is blocked (JB) so the panel of Kt
 * (or V) that the inner loops sweep stays in L1/L2 while all rows of Q go
 * past it. Softmax is the same as attn_loops.c.
 */
#define _POSIX_C_SOURCE 200809L
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
#define MR 4   /* rows of the register tile               */
#define NR 16  /* columns of the register tile (2 x ymm)  */
#define JB 128 /* cache block along j: Kt panel = d*JB floats */

/* S[i..i+MR) [j..j+NR) = Qs[i..] * Kt[:, j..]   (full tile) */
static inline void qk_tile(const float *restrict Qs, const float *restrict Kt,
                           float *restrict S, int N, int d, int i, int j)
{
    float acc[MR][NR] = {{0}};
    for (int k = 0; k < d; k++) {
        const float *restrict kt = Kt + (size_t)k * N + j;
        for (int r = 0; r < MR; r++) {
            const float q = Qs[(size_t)(i + r) * d + k];
            for (int c = 0; c < NR; c++)
                acc[r][c] += q * kt[c];
        }
    }
    for (int r = 0; r < MR; r++)
        for (int c = 0; c < NR; c++)
            S[(size_t)(i + r) * N + j + c] = acc[r][c];
}

/* O[i..i+MR) [c0..c0+NR) += S[i.., j0..j1) * V[j0..j1, c0..] (full tile) */
static inline void sv_tile(const float *restrict S, const float *restrict V,
                           float *restrict O, int N, int d, int i, int c0, int j0, int j1)
{
    float acc[MR][NR];
    for (int r = 0; r < MR; r++)
        for (int c = 0; c < NR; c++)
            acc[r][c] = O[(size_t)(i + r) * d + c0 + c];
    for (int j = j0; j < j1; j++) {
        const float *restrict v = V + (size_t)j * d + c0;
        for (int r = 0; r < MR; r++) {
            const float s = S[(size_t)(i + r) * N + j];
            for (int c = 0; c < NR; c++)
                acc[r][c] += s * v[c];
        }
    }
    for (int r = 0; r < MR; r++)
        for (int c = 0; c < NR; c++)
            O[(size_t)(i + r) * d + c0 + c] = acc[r][c];
}

void attention(const float *Q, const float *K, const float *V, float *O, int N, int d)
{
    const float scale = 1.0f / sqrtf((float)d);
    float *S = malloc((size_t)N * N * sizeof(float));
    float *Kt = malloc((size_t)d * N * sizeof(float));
    float *Qs = malloc((size_t)N * d * sizeof(float));

    for (int j = 0; j < N; j++)
        for (int k = 0; k < d; k++)
            Kt[(size_t)k * N + j] = K[(size_t)j * d + k];
    for (size_t x = 0; x < (size_t)N * d; x++)
        Qs[x] = Q[x] * scale;

    const int Nm = N - N % MR, Nn = N - N % NR, dn = d - d % NR;

    /* 1. S = Qs Kt */
    for (int jj = 0; jj < Nn; jj += JB) {
        const int jend = jj + JB < Nn ? jj + JB : Nn;
        for (int i = 0; i < Nm; i += MR)
            for (int j = jj; j < jend; j += NR)
                qk_tile(Qs, Kt, S, N, d, i, j);
    }
    /* edges (rows i >= Nm or columns j >= Nn), never hit when N % 16 == 0 */
    for (int i = 0; i < N; i++)
        for (int j = (i < Nm ? Nn : 0); j < N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < d; k++)
                acc += Qs[(size_t)i * d + k] * Kt[(size_t)k * N + j];
            S[(size_t)i * N + j] = acc;
        }

    /* 2. softmax of each row of S */
    for (int i = 0; i < N; i++) {
        float *restrict row = S + (size_t)i * N;
        float max = row[0];
        for (int j = 1; j < N; j++)
            max = row[j] > max ? row[j] : max;
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            row[j] = expf(row[j] - max);
            sum += row[j];
        }
        const float inv = 1.0f / sum;
        for (int j = 0; j < N; j++)
            row[j] *= inv;
    }

    /* 3. O = S V */
    memset(O, 0, (size_t)N * d * sizeof(float));
    for (int jj = 0; jj < N; jj += JB) {
        const int jend = jj + JB < N ? jj + JB : N;
        for (int i = 0; i < Nm; i += MR)
            for (int c = 0; c < dn; c += NR)
                sv_tile(S, V, O, N, d, i, c, jj, jend);
    }
    for (int i = 0; i < N; i++)
        for (int c = (i < Nm ? dn : 0); c < d; c++) {
            float acc = 0.0f;
            for (int j = 0; j < N; j++)
                acc += S[(size_t)i * N + j] * V[(size_t)j * d + c];
            O[(size_t)i * d + c] = acc;
        }

    free(Qs);
    free(Kt);
    free(S);
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
