int main(int argc, char** argv)
{
  /* check OMP use
  int option = openblas_get_parallel();
  printf("the option is : %d\n", option); // thus using openMP
  */

  clock_t begin = clock();

  int k , i , j , sj, si, sk, nrep = 1 , cnt = 0;
  float *A , *B , *C;
  if ( argc != 5 )
  {
    puts("Format: ./a.out number_of_iteration dimension_1 dimension_2 dimension_3");
    exit(0);
  }
  if ( argc == 5 )
  {
    sscanf ( argv[1] , "%d" , &i );
    sscanf ( argv[2] , "%d" , &k );
    sscanf ( argv[3] , "%d" , &j );
  }
  A = (float *) malloc( sizeof(float) * i * k );
  B = (float *) malloc( sizeof(float) * k * j );
  C = (float *) malloc( sizeof(float) * i * j );
  init_matrix ( A , i , k );
  init_matrix ( B , k , j );

  omp_set_num_threads(4);

  #pragma omp parallel shared(A,B,C, i, k, j) private(si, sk, sj)   // add more!
  { // must be in a newline
  #pragma omp for  // sj,si,sk declare outsize // what loop to parallelize, can parallelize multiple loops simulatenously?
  int threads = omp_get_num_threads();
  printf("num of threads: %d\n", threads);
  for (sj = 0; sj < j; sj += BLOCKSIZE){
    for (si = 0; si < i; si += BLOCKSIZE) {
      for (sk = 0; sk < k; sk += BLOCKSIZE) { //cut, cut, cut all dimensions
	do_block(si, sj, sk, A, B, C, i, k, j);
	//Tiled_SGEMM, pass starting point for all dimension, and add as well.
      }
    }
  }
  } // end of omp parallel
  print_start_of_Matrice(C, j);

  clock_t end = clock();
  double time_spent = (double)(end-begin)/CLOCKS_PER_SEC;
  printf("time spent: %f\n",time_spent);

  return 0;
}
