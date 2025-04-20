#include "../helper.h"
#include <cuda.h>
#include "CGBN/cgbn.h"

// IMPORTANT:  DO NOT DEFINE TPI OR BITS BEFORE INCLUDING CGBN
#define TPI  32 // at least 8 words per thread 
#define BITS (128*32)//2048 //3200

#include "cgbn-kers.cu.h"

#define GPU_RUNS_ADD  500
#define GPU_RUNS_CMUL 500
#define GPU_RUNS_DIV 500
#define GPU_RUNS_POLY 125

/****************************/
/***  support routines    ***/
/****************************/

void cgbn_check(cgbn_error_report_t *report, const char *file=NULL, int32_t line=0) {
  // check for cgbn errors
  if(cgbn_error_report_check(report)) {
    printf("\n");
    printf("CGBN error occurred: %s\n", cgbn_error_string(report));

    if(report->_instance!=0xFFFFFFFF) {
      printf("Error reported by instance %d", report->_instance);
      if(report->_blockIdx.x!=0xFFFFFFFF || report->_threadIdx.x!=0xFFFFFFFF)
        printf(", ");
      if(report->_blockIdx.x!=0xFFFFFFFF)
      printf("blockIdx=(%d, %d, %d) ", report->_blockIdx.x, report->_blockIdx.y, report->_blockIdx.z);
      if(report->_threadIdx.x!=0xFFFFFFFF)
        printf("threadIdx=(%d, %d, %d)", report->_threadIdx.x, report->_threadIdx.y, report->_threadIdx.z);
      printf("\n");
    }
    else {
      printf("Error reported by blockIdx=(%d %d %d)", report->_blockIdx.x, report->_blockIdx.y, report->_blockIdx.z);
      printf("threadIdx=(%d %d %d)\n", report->_threadIdx.x, report->_threadIdx.y, report->_threadIdx.z);
    }
    if(file!=NULL)
      printf("file %s, line %d\n", file, line);
    exit(1);
  }
}
#define CGBN_CHECK(report) cgbn_check(report, __FILE__, __LINE__)

// BITS = 16384 

// support routine to generate random instances
instance_t *generate_instances(uint32_t count) {
  instance_t *instances=(instance_t *)malloc(sizeof(instance_t)*count);
  printf("BITS = %d, nz = %d \n", BITS, (BITS/32));
  for(int index=0;index<count;index++) {  
    ourMkRandom<BITS/32, BITS/32>(1, instances[index].a._limbs);
    ourMkRandom<BITS/32, BITS/32>(1, instances[index].b._limbs);
    //random_words(instances[index].a._limbs, BITS/32);
    //random_words(instances[index].b._limbs, BITS/32);
  }
  return instances;
}

// support routine to generate random instances
instance_t_div *generate_instances_div(uint32_t count) {
  instance_t_div *instances=(instance_t_div *)malloc(sizeof(instance_t_div)*count);
  printf("BITS = %d, nz = %d \n", BITS, (BITS/32));
  for(int index=0;index<count;index++) {  
    ourMkRandom<BITS/32, BITS/32>(1, instances[index].a._limbs);
    ourMkRandom<BITS/32, BITS/32>(1, instances[index].b._limbs);
    //random_words(instances[index].a._limbs, BITS/32);
    //random_words(instances[index].b._limbs, BITS/32);
  }
  return instances;
}

void verifyResults(bool is_add, uint32_t num_instances, instance_t  *instances) {
    uint32_t buffer[BITS/32];
    for(uint32_t i=0; i<num_instances; i++) {
        gmpAddMulOnce<BITS/32>(is_add, &instances[i].a._limbs[0], &instances[i].b._limbs[0], &buffer[0]);
        for(uint32_t j=0; j<BITS/32; j++) {
             if ( buffer[j] != instances[i].sum._limbs[j] ) {
                printf( "INVALID RESULT at instance: %u, local index %u: %u vs %u\n"
                      , i, j, buffer[j], instances[i].sum._limbs[j]
                      );
                return;
            }
        }
    }
    printf("VALID!\n");
}

void verifyResultsQuo(uint32_t num_instances, instance_t  *instances) {
    uint32_t buffer[BITS/32];
    for(uint32_t i=0; i<num_instances; i++) {
        gmpQuoOnce<uint32_t, BITS/32>(&instances[i].a._limbs[0], &instances[i].b._limbs[0], &buffer[0]);
        for(uint32_t j=0; j<BITS/32; j++) {
             if ( buffer[j] != instances[i].sum._limbs[j] ) {
                printf( "INVALID RESULT at instance: %u, local index %u: %u vs %u\n"
                      , i, j, buffer[j], instances[i].sum._limbs[j]
                      );
                return;
            }
        }
    }
    printf("VALID!\n");
}

void verifyResults_div(uint32_t num_instances, instance_t_div  *instances) {
  printf("lol?");
  uint32_t buffer[BITS/32];
  uint32_t buffer2[BITS/32];
  for(uint32_t i=0; i<num_instances; i++) {
    printf("sker der?");
      gmpDivOnce<uint32_t, BITS/32>(&instances[i].a._limbs[0], &instances[i].b._limbs[0], &buffer[0], &buffer2[0]);
      for(uint32_t j=0; j<BITS/32; j++) {
           if ( (buffer[j] != instances[i].quo._limbs[j]) || (buffer2[j] != instances[i].rem._limbs[j]) ) {
              printf( "INVALID RESULT at instance: %u, local index %u: %u vs %u\n"
                    , i, j, buffer[j], instances[i].quo._limbs[j]
                    );
              return;
          }
      }
  }
  printf("VALID!\n");
}

void runAdd ( const uint32_t num_instances, const uint32_t cuda_block
            , cgbn_error_report_t *report,  instance_t  *gpuInstances
            , instance_t  *instances
) {
    //printf("Running GPU kernel ...\n");

    const uint32_t ipb = cuda_block/TPI;

	// start timer
	unsigned long int elapsed = 0;
	struct timeval t_start, t_end, t_diff;
	gettimeofday(&t_start, NULL);

	// launch with 32 threads per instance, 128 threads (4 instances) per block
	for(int i = 0; i < GPU_RUNS_ADD; i++)
		kernel_add<<<(num_instances+ipb-1)/ipb, cuda_block>>>(report, gpuInstances, num_instances);
	cudaDeviceSynchronize();
	
	//end timer
	gettimeofday(&t_end, NULL);
	timeval_subtract(&t_diff, &t_end, &t_start);
	elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS_ADD;
	
	//printf("Average of %d runs: %ld\n", GPU_RUNS_ADD, elapsed);
	
	gpuAssert( cudaPeekAtLastError() );

    const uint32_t m = BITS / 32;
    double runtime_microsecs = elapsed; 
    double bytes_accesses = 3.0 * num_instances * m * sizeof(uint32_t);  
    double gigabytes = bytes_accesses / (runtime_microsecs * 1000);

    printf( "CGBN Addition (num-instances = %d, num-word-len = %d, total-size: %d) \
runs in: %lu microsecs, GB/sec: %.2f, Mil-Instances/sec: %.2f\n"
          , num_instances, m, num_instances * m, elapsed
          , gigabytes, num_instances / runtime_microsecs
          );
	
    // error report uses managed memory, so we sync the device (or stream) and check for cgbn errors
	CUDA_CHECK(cudaDeviceSynchronize());
	CGBN_CHECK(report);

	// copy the instances back from gpuMemory
	//printf("Copying results back to CPU ...\n");
	CUDA_CHECK(cudaMemcpy(instances, gpuInstances, sizeof(instance_t)*num_instances, cudaMemcpyDeviceToHost));

	//printf("Verifying the results ...\n");
	//verify_results(instances, num_instances);
	verifyResults(true, num_instances, instances);
	
	{ // testing 6 additions // kernel_6adds
	    unsigned long int elapsed = 0;
	    struct timeval t_start, t_end, t_diff;
	    gettimeofday(&t_start, NULL);

	    // launch with 32 threads per instance, 128 threads (4 instances) per block
	    for(int i = 0; i < GPU_RUNS_ADD; i++)
		    kernel_add<<<(num_instances+ipb-1)/ipb, cuda_block>>>(report, gpuInstances, num_instances);
	    cudaDeviceSynchronize();
	    
	    //end timer
	    gettimeofday(&t_end, NULL);
	    timeval_subtract(&t_diff, &t_end, &t_start);
	    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS_ADD;
	    
	    //printf("Average of %d runs: %ld\n", GPU_RUNS_ADD, elapsed);
	    
	    gpuAssert( cudaPeekAtLastError() );

        const uint32_t m = BITS / 32;
        double runtime_microsecs = elapsed; 
        double bytes_accesses = 3.0 * num_instances * m * sizeof(uint32_t);  
        double gigabytes = bytes_accesses / (runtime_microsecs * 1000);

        printf( "CGBN SIX Additions (num-instances = %d, num-word-len = %d, total-size: %d) \
runs in: %lu microsecs, GB/sec: %.2f, Mil-Instances/sec: %.2f\n"
              , num_instances, m, num_instances * m, elapsed
              , gigabytes, num_instances / runtime_microsecs
              );
	    
        // error report uses managed memory, so we sync the device (or stream) and check for cgbn errors
	    CUDA_CHECK(cudaDeviceSynchronize());
	    CGBN_CHECK(report);
	}
}

void runMul ( const uint32_t num_instances, const uint32_t cuda_block
            , cgbn_error_report_t *report,  instance_t  *gpuInstances
            , instance_t  *instances
) {
    //printf("Running GPU kernel ...\n");

  const uint32_t ipb = cuda_block/TPI;

	// start timer
	unsigned long int elapsed = 0;
	struct timeval t_start, t_end, t_diff;
	gettimeofday(&t_start, NULL);

	// launch with 32 threads per instance, 128 threads (4 instances) per block
	for(int i = 0; i < GPU_RUNS_CMUL; i++)
		kernel_mul<<<(num_instances+ipb-1)/ipb, cuda_block>>>(report, gpuInstances, num_instances);
	cudaDeviceSynchronize();
	
	//end timer
	gettimeofday(&t_end, NULL);
	timeval_subtract(&t_diff, &t_end, &t_start);
	elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS_CMUL;
	
	//printf("Average of %d runs: %ld\n", GPU_RUNS_CMUL, elapsed);
	
	gpuAssert( cudaPeekAtLastError() );

    const uint32_t m = BITS / 32;
    double runtime_microsecs = elapsed;
    //double num_u32_ops = 4.0 * num_instances * m * m; 
    double num_u32_ops = num_instances * numAd32OpsOfMultInst<uint32_t>(m);
    double gigaopsu32 = num_u32_ops / (runtime_microsecs * 1000);

    printf( "CGBN Multiply (num-instances = %d, num-word-len = %d, total-size: %d), \
averaged over %d runs: %lu microsecs, Gopsu32/sec: %.2f, Mil-Instances/sec: %.2f\n"
          , num_instances, m, num_instances * m, GPU_RUNS_CMUL
          , elapsed, gigaopsu32, num_instances / runtime_microsecs
          );
	
    // error report uses managed memory, so we sync the device (or stream) and check for cgbn errors
	CUDA_CHECK(cudaDeviceSynchronize());
	CGBN_CHECK(report);

	// copy the instances back from gpuMemory
	// printf("Copying results back to CPU, size of instance_t: %d ...\n", sizeof(instance_t));
	CUDA_CHECK(cudaMemcpy(instances, gpuInstances, sizeof(instance_t)*num_instances, cudaMemcpyDeviceToHost));

	//printf("Verifying the results ...\n");
	//verify_results(instances, num_instances);
	verifyResults(false, num_instances, instances);
}

void runPoly( const uint32_t num_instances, const uint32_t cuda_block
            , cgbn_error_report_t *report,  instance_t  *gpuInstances
            , instance_t  *instances
) {
    //printf("Running GPU kernel ...\n");

    const uint32_t ipb = cuda_block/TPI;

	// start timer
	unsigned long int elapsed = 0;
	struct timeval t_start, t_end, t_diff;
	gettimeofday(&t_start, NULL);

	// launch with 32 threads per instance, 128 threads (4 instances) per block
	for(int i = 0; i < GPU_RUNS_POLY; i++)
		kernel_poly<<<(num_instances+ipb-1)/ipb, cuda_block>>>(report, gpuInstances, num_instances);
	cudaDeviceSynchronize();
	
	//end timer
	gettimeofday(&t_end, NULL);
	timeval_subtract(&t_diff, &t_end, &t_start);
	elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS_POLY;
	
	//printf("Average of %d runs: %ld\n", GPU_RUNS_POLY, elapsed);
	
	gpuAssert( cudaPeekAtLastError() );

    const uint32_t m = BITS / 32;
    double runtime_microsecs = elapsed;
    //double num_u32_ops = 4.0 * 4.0 * num_instances * m * m;
    double num_u32_ops = 4.0 * num_instances * numAd32OpsOfMultInst<uint32_t>(m);
    double gigaopsu32 = num_u32_ops / (runtime_microsecs * 1000);

    printf( "CGBN Polynomial (num-instances = %d, num-word-len = %d, total-size: %d), \
averaged over %d runs: %lu microsecs, Gopsu32/sec: %.2f, Mil-Instances/sec: %.2f\n"
          , num_instances, m, num_instances * m, GPU_RUNS_POLY
          , elapsed, gigaopsu32, num_instances / runtime_microsecs
          );
	
    // error report uses managed memory, so we sync the device (or stream) and check for cgbn errors
	CUDA_CHECK(cudaDeviceSynchronize());
	CGBN_CHECK(report);

	// copy the instances back from gpuMemory
	//printf("Copying results back to CPU ...\n");
	CUDA_CHECK(cudaMemcpy(instances, gpuInstances, sizeof(instance_t)*num_instances, cudaMemcpyDeviceToHost));

	//printf("Verifying the results ...\n");
	//verify_results(instances, num_instances);
}

void runDiv ( const uint32_t num_instances, const uint32_t cuda_block
  , cgbn_error_report_t *report,  instance_t_div  *gpuInstances
  , instance_t_div  *instances
) {
  //printf("Running GPU kernel ...\n");

  const uint32_t ipb = cuda_block/TPI;

  // start timer
  unsigned long int elapsed = 0;
  struct timeval t_start, t_end, t_diff;
  gettimeofday(&t_start, NULL);

  // launch with 32 threads per instance, 128 threads (4 instances) per block
  for(int i = 0; i < GPU_RUNS_DIV; i++){
    kernel_div<<<(num_instances+ipb-1)/ipb, cuda_block>>>(report, gpuInstances, num_instances);
  }
  cudaDeviceSynchronize();

  //end timer
  gettimeofday(&t_end, NULL);
  timeval_subtract(&t_diff, &t_end, &t_start);
  elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS_DIV;

  //printf("Average of %d runs: %ld\n", GPU_RUNS_DIV, elapsed);

  gpuAssert( cudaPeekAtLastError() );

  const uint32_t m = BITS / 32;
  double runtime_microsecs = elapsed;
  //double num_u32_ops = 4.0 * num_instances * m * m; 
  double num_u32_ops = num_instances * numAd32OpsOfMultInst<uint32_t>(m);
  double gigaopsu32 = num_u32_ops / (runtime_microsecs * 1000);

  printf( "CGBN division (num-instances = %d, num-word-len = %d, total-size: %d), \
averaged over %d runs: %lu microsecs, Gopsu32/sec: %.2f, Mil-Instances/sec: %.2f\n"
    , num_instances, m, num_instances * m, GPU_RUNS_DIV
    , elapsed, gigaopsu32, num_instances / runtime_microsecs
  );

  // error report uses managed memory, so we sync the device (or stream) and check for cgbn errors
  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);

  // copy the instances back from gpuMemory
  // printf("Copying results back to CPU, size of instance_t: %d ...\n", sizeof(instance_t));
  CUDA_CHECK(cudaMemcpy(instances, gpuInstances, sizeof(instance_t_div)*num_instances, cudaMemcpyDeviceToHost));

  //printf("Verifying the results ...\n");
  //verify_results(instances, num_instances);

  // verifyResults_div causes segmentation fault inside gmpDivOnce, fix incomming.
  //verifyResults_div(num_instances, instances);
}

void runQuo ( const uint32_t num_instances, const uint32_t cuda_block
  , cgbn_error_report_t *report,  instance_t  *gpuInstances
  , instance_t  *instances
) {
  //printf("Running GPU kernel ...\n");

  const uint32_t ipb = cuda_block/TPI;

  // start timer
  unsigned long int elapsed = 0;
  struct timeval t_start, t_end, t_diff;
  gettimeofday(&t_start, NULL);

  // launch with 32 threads per instance, 128 threads (4 instances) per block
  for(int i = 0; i < GPU_RUNS_DIV; i++){
    kernel_quo<<<(num_instances+ipb-1)/ipb, cuda_block>>>(report, gpuInstances, num_instances);
  }
  cudaDeviceSynchronize();

  //end timer
  gettimeofday(&t_end, NULL);
  timeval_subtract(&t_diff, &t_end, &t_start);
  elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS_DIV;

  //printf("Average of %d runs: %ld\n", GPU_RUNS_DIV, elapsed);

  gpuAssert( cudaPeekAtLastError() );

  const uint32_t m = BITS / 32;
  double runtime_microsecs = elapsed;
  //double num_u32_ops = 4.0 * num_instances * m * m; 
  double num_u32_ops = num_instances * numAd32OpsOfMultInst<uint32_t>(m);
  double gigaopsu32 = num_u32_ops / (runtime_microsecs * 1000);

  printf( "CGBN Quotient (num-instances = %d, num-word-len = %d, total-size: %d), \
averaged over %d runs: %lu microsecs, Gopsu32/sec: %.2f, Mil-Instances/sec: %.2f\n"
    , num_instances, m, num_instances * m, GPU_RUNS_DIV
    , elapsed, gigaopsu32, num_instances / runtime_microsecs
  );

  // error report uses managed memory, so we sync the device (or stream) and check for cgbn errors
  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);

  // copy the instances back from gpuMemory
  // printf("Copying results back to CPU, size of instance_t: %d ...\n", sizeof(instance_t));
  CUDA_CHECK(cudaMemcpy(instances, gpuInstances, sizeof(instance_t)*num_instances, cudaMemcpyDeviceToHost));


  //printf("Verifying the results ...\n");
  //verify_results(instances, num_instances);
  verifyResultsQuo(num_instances, instances);
}


int main(int argc, char * argv[]) {
    if (argc != 2) {
        printf("Usage: %s <number-of-instances>\n", argv[0]);
        exit(1);
    }
        
    const int num_instances = atoi(argv[1]);
    
    instance_t          *instances, *gpuInstances;
    instance_t_div      *instances_div, *gpuInstances_div; 
	cgbn_error_report_t *report;

	//printf("Genereating instances ...\n");
	instances=generate_instances(num_instances);
  instances_div=generate_instances_div(num_instances);
  //printf("%d",num_instances);
  //for (int i = 0; i < num_instances; i++){
  //  //printInstance(i, instances);
  //  for (int j = 0; j < (BITS/32); j++){
  //  printf("%u & %u @ (%d,%d) \n", instances[i].a._limbs[j], instances[i].b._limbs[j], i, j);
  //  }
  //}

	//printf("Copying instances to the GPU ...\n");
	CUDA_CHECK(cudaSetDevice(0));
	CUDA_CHECK(cudaMalloc((void **)&gpuInstances, sizeof(instance_t)*num_instances));
  CUDA_CHECK(cudaMalloc((void **)&gpuInstances_div, sizeof(instance_t_div)*num_instances));
	CUDA_CHECK(cudaMemcpy(gpuInstances, instances, sizeof(instance_t)*num_instances, cudaMemcpyHostToDevice));
  
	// create a cgbn_error_report for CGBN to report back errors
	CUDA_CHECK(cgbn_error_report_alloc(&report)); 
  
  
  runAdd (num_instances, 128, report, gpuInstances, instances);
  runMul (num_instances, 128, report, gpuInstances, instances);
  // runPoly(num_instances, 128, report, gpuInstances, instances);
  runQuo(num_instances, 128, report, gpuInstances, instances);
  
  // clean up
  free(instances);
  CUDA_CHECK(cudaFree(gpuInstances));
  
  CUDA_CHECK(cudaMemcpy(gpuInstances_div, instances_div, sizeof(instance_t_div)*num_instances, cudaMemcpyHostToDevice));
    
  runDiv(num_instances, 128, report, gpuInstances_div, instances_div);

  free(instances_div);
  CUDA_CHECK(cudaFree(gpuInstances_div));

	CUDA_CHECK(cgbn_error_report_free(report));

    return 0;
}
