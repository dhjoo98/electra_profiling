#include "./headers/cblas.h"
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

void init_matrix(float* A, int dim1 , int dim2 ) // (num of row, num of col)
{
  int mod = 100003, prod = 7 , e = 1 , i = 0, j = 0;
  for ( i = 0; i < dim1; i++ )
  {
	for ( j = 0; j < dim2; j++ )
        {
            e = (e*prod + 1)%mod; // random
            A[i*dim2 + j] = e * .91739210437;
        }
  }
}

void print_start_of_Matrice(float* A, int dim2)
{
  int row, column;
  for (row=90; row<95;row++){
    for (column=90; column<95;column++){
	printf("%f\t", A[row*dim2 + column]);
    }
  printf("\n");
  }
}

#define BLOCKSIZE 10
void do_block (int bi, int bj, int bk, float *A, float *B, float *C, int len_i, int len_k, int len_j) {
  int i,k,j;
  for (i=0; i<BLOCKSIZE; i++){
    for (j=0; j<BLOCKSIZE; j++){
      for (k=0; k<BLOCKSIZE; k++){
        C[(bi+i)*len_j + bj + j] += A[(bi+i)*len_k + bk + k ]*B[(bk+k)*len_j + bj + j];
}}}}

int main(int argc, char** argv)
{
  /* check OMP use
  int option = openblas_get_parallel();
  printf("the option is : %d\n", option); // thus using openMP
  */



  int len_k , len_i , len_j , bj, bi, bk;
  float *A , *B , *C;
  if ( argc != 4 )
  {
    puts("Format: ./a.out number_of_iteration dimension_1 dimension_2 dimension_3");
    exit(0);
  }
  if ( argc == 4 )
  {
    sscanf ( argv[1] , "%d" , &len_i );
    sscanf ( argv[2] , "%d" , &len_k );
    sscanf ( argv[3] , "%d" , &len_j );
  }
  A = (float *) malloc( sizeof(float) * len_i * len_k );
  B = (float *) malloc( sizeof(float) * len_k * len_j );
  C = (float *) malloc( sizeof(float) * len_i * len_j );
  //printf("dimension: %d,%d,%d\n", len_i, len_k, len_j); come on!???
  init_matrix ( A , len_i , len_k );
  init_matrix ( B , len_k , len_j );

  //omp_set_num_threads(2);
clock_t begin = clock();
  #pragma omp parallel shared(A,B,C, len_i, len_k, len_j) private(bi, bk, bj)
  {
  #pragma omp for
  for (bi = 0; bi < len_i; bi += BLOCKSIZE){
    for (bj = 0; bj < len_j; bj += BLOCKSIZE) {
      for (bk = 0; bk < len_k; bk += BLOCKSIZE) {
	do_block(bi, bj, bk, A, B, C, len_i, len_k, len_j); //matmul for the small matrices
  }}}
    int threads = omp_get_num_threads();
    printf("num of threads: %d\n", threads);
  } // end of omp parallel
clock_t end = clock();

  printf("----------------------------\n");
  print_start_of_Matrice(A, len_k);
  printf("----------------------------\n");
  print_start_of_Matrice(B, len_j);
  printf("----------------------------\n");
  print_start_of_Matrice(C, len_j);


  double time_spent = (double)(end-begin)/CLOCKS_PER_SEC;
  printf("time spent: %f\n",time_spent);

  return 0;
}
