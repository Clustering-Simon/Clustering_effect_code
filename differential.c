/*
  Accompanying code to the paper "Clustering Effect in Simon and Simeck"

  Computes a lower boud on the probability of a differential for Simon or Simeck

  Compile with gcc -Wall -Wextra -O3 -march=native -fopenmp -lm
 */

#define _GNU_SOURCE
#define _XOPEN_SOURCE 500

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <assert.h>

#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>

// Parameters
#define SIMECK
// #define SIMON
#define PRECISION 14 // Window size (w)
#define ROUNDS 40

// #define PRINT_DISTRIBUTION
#define PRINT_MAX

// mmap-based memory allocation reduces memory usage for Simon
#define USE_MMAP

#if defined(SIMON) == defined(SIMECK)
#error Please only exactly one of SIMON and SIMECK
#endif


#define ROT(x,n) ( ((uint32_t)(x)<<((n)&31)) | ((uint32_t)(x)>>((32-(n))&31)) )

uint32_t f(uint32_t x) {
#ifdef SIMECK
  return (ROT(x, 5)&x)^ROT(x,1); // Simeck
#endif
#ifdef SIMON
  return (ROT(x, 8)&ROT(x, 1))^ROT(x,2); // Simon
#endif
}

typedef double distribution[1<<PRECISION][1<<PRECISION];

typedef struct {
  uint32_t B;
  uint32_t A[PRECISION]; // vector space AX+B
  double proba;
} transition_space;


#ifdef USE_MMAP

// mmap-based allocation can overcommit and leave holes in allocation
// This reduces memory usage for Simon
void *alloc_distribution() {
  void *p = mmap(NULL, sizeof(distribution), PROT_READ | PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0);
  assert(p != MAP_FAILED);
  // NOHUGEPAGE reduces memory usage but reduces performance
  // madvise(p, sizeof(distribution), MADV_NOHUGEPAGE);
  return p;
}

void free_distribution(void *p) {
  munmap(p, sizeof(distribution));
}

#else

void *alloc_distribution() {
  void *p = calloc(1, sizeof(distribution));
  assert(p);
  return p;
}

void free_distribution(void *p) {
  free(p);
}

#endif

void round_trans(distribution in, distribution out, transition_space *m) {
#pragma omp parallel for schedule(dynamic)
  // NOTE: left are right are swapped to optimize memory access
  for (int l=0; l<1<<PRECISION; l++) {
    if (m[l].B < 1<<PRECISION) {
      for (int r=0; r<1<<PRECISION; r++) {
	// Generate output space with Gray enumeration
	if (in[r][l]) {
	  uint32_t d = m[l].B;
	  for (int i=1;; i++) {
	    out[l][d^r] += in[r][l]*m[l].proba;
	    int z = __builtin_ctz(i);
	    if (z<PRECISION && m[l].A[z])
	      d ^= m[l].A[z];
	    else
	      break;
	  }
	}
      }
    }
  }
}

transition_space* init_transition() {
  transition_space* matrix = calloc(1<<PRECISION, sizeof(transition_space));
  
  for (int delta=0; delta < 1<<PRECISION; delta++) {
    // Compute vector space
    uint32_t B = f(0)^f(delta);
    uint32_t X[32];
    for (int i=0; i<32; i++) {
      X[i] = f(1<<i)^f(delta^(1<<i))^B;
    }
    // Gaussian reduce
    int p = 31;
    for (int i=0; i<32; i++) {
      // Find pivot
      while (p>=0 && (X[i] & (1<<p)) == 0) {
	for (int j=i+1; j<31; j++) {
	  if (X[j] & (1<<p)) {
	    X[i] ^= X[j];
	    goto PIVOT;;
	  }
	}
	p--;
      }
    PIVOT:
      if (p<0)
	break;
      // Reduce
      for (int j=i+1; j<32; j++) {
	if (X[j] & (1<<p)) X[j] ^= X[i];
      }
      if (B & (1<<p)) B ^= X[i];
    }

    matrix[delta].proba = 1;
    matrix[delta].B = B;
    int j = 0;
    for (int i=31; i>=0; i--)
      if (X[i])	{
	matrix[delta].proba /= 2;
	if (X[i] < 1<<PRECISION) {
	  matrix[delta].A[j++] = X[i];
	}
      }
  }
  return matrix;
}

int main() {
  transition_space *m = init_transition();

  double (*d)[1<<PRECISION] = alloc_distribution();
  // Input difference (0,1)
  d[1][0] = 1;
  // d[2][1] = 1;
 
  for (int i=1; i<ROUNDS; i++) {
#ifdef PRINT_DISTRIBUTION
    {
      // Print distribution
      for (int i=0; i<257; i++) {
	printf ("[%2i]: ", i);
	for (int j=0; j<257; j++) {
	  double m = log2(d[i][j]);
	  printf (" %6.2f", m);
	}
	printf ("\n");
      }
    }
#endif
#ifdef PRINT_MAX
    {
      // Print max
      if (i == 1) {
	printf ("Input:\n");
      }
      double max = 0;
#pragma omp parallel for reduction(max: max)
      for (int i=0; i < 1<<PRECISION; i++) {
	for (int j=0; j < 1<<PRECISION; j++) {
	  if (d[i][j]>max) {
	    max = d[i][j];
	  }
	}
      }
      printf ("Max: %f", log2(max));
#pragma omp parallel for
      for (int i=0; i < 1<<PRECISION; i++) {
	for (int j=0; j < 1<<PRECISION; j++) {
	  if (d[i][j] > max*0.999999) {
#pragma omp critical
	    printf (" (%i,%i)", j, i);
	  }
	}
      }
      printf ("\n");
    }
#endif
    fflush(stdout);
    double (*tmp)[1<<PRECISION] = alloc_distribution();
    round_trans(d, tmp, m);
    // Output difference (1,0)
    printf("Round %2i: %f\n", i, log2(tmp[0][1]));
    // printf("Round %2i: %f\n", i, log2(tmp[1][2]));
    fflush(stdout);
    free_distribution(d);
    d = tmp;
  }
}
