#include "../helper.h"
#include <cuda.h>
#include "CGBN/cgbn.h"

// IMPORTANT:  DO NOT DEFINE TPI OR BITS BEFORE INCLUDING CGBN
#define TPI  THD_PER_INST //THD_per_inst // at least 8 words per thread 
#define BITS (NUM_BITS) //(NUM_BITS)//2048 //3200

#include "cgbn-kers.cu.h"

#define GPU_RUNS_ADD  500
#define GPU_RUNS_CMUL 200
#define GPU_RUNS_DIV  200
#define GPU_RUNS_POLY 125
#define GPU_RUNS_GCD  50

// Combination of u and v results in 6 iterations of the GCD algorithm
#define GCD_U_VAL 46 
#define GCD_V_VAL 28


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
    ourMkRandom<BITS/32, BITS/32-8>(1, instances[index].a._limbs);
    ourMkRandom<BITS/32, 3>(1, instances[index].b._limbs);
    //random_words(instances[index].a._limbs, BITS/32);
    //random_words(instances[index].b._limbs, BITS/32);
  }
  return instances;
}

// support routine to generate random instances
instance_div_t *generate_instances_div(uint32_t count) {
  instance_div_t *instances=(instance_div_t *)malloc(sizeof(instance_div_t)*count);
  printf("BITS = %d, nz = %d \n", BITS, (BITS/32));
  for(int index=0;index<count;index++) {  
    ourMkRandom<BITS/32, BITS/32-8>(1, instances[index].a._limbs);
    ourMkRandom<BITS/32, (BITS/32)/2-3>(1, instances[index].b._limbs);
    //random_words(instances[index].a._limbs, BITS/32);
    //random_words(instances[index].b._limbs, BITS/32);
  }
  return instances;
}

instance_t *generate_instances_gcd(uint32_t count) {
  instance_t *instances=(instance_t *)malloc(sizeof(instance_t)*count);
  printf("BITS = %d, nz = %d \n", BITS, (BITS/32));
  for(int index=0;index<count;index++) {  
    ourMkSet<BITS/32>(1, instances[index].a._limbs, GCD_U_VAL, (BITS/32)-1);
    ourMkSet<BITS/32>(1, instances[index].b._limbs, GCD_V_VAL, (BITS/32)-1);
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


void verifyResultsGCD(uint32_t num_instances, instance_t  *instances) {
    uint32_t buffer[BITS/32];
    for(uint32_t i=0; i<num_instances; i++) {
        gmpGCDOnce<uint32_t, BITS/32>(&instances[i].a._limbs[0], &instances[i].b._limbs[0], &buffer[0]);
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

void verifyResultsDiv(uint32_t num_instances, instance_div_t  *instances) {
    uint32_t bufferQuo[BITS/32];
    uint32_t bufferRem[BITS/32];
    for(uint32_t i=0; i<num_instances; i++) {
        gmpDivOnce<uint32_t, BITS/32>(&instances[i].a._limbs[0], &instances[i].b._limbs[0], &bufferQuo[0], &bufferRem[0]);
        for(uint32_t j=0; j<BITS/32; j++) {
             if ( bufferQuo[j] != instances[i].quo._limbs[j] ) {
                printf( "INVALID RESULT at quotient: %u, local index %u: %u vs %u\n"
                      , i, j, bufferQuo[j], instances[i].quo._limbs[j]
                      );
                return;
            } else if ( bufferRem[j] != instances[i].rem._limbs[j] ) {
                printf( "INVALID RESULT at remainder: %u, local index %u: %u vs %u\n"
                      , i, j, bufferRem[j], instances[i].rem._limbs[j]
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
    double num_u32_ops = num_instances * numAd32OpsOfClassicalMultInst<uint32_t>(m);
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
  , cgbn_error_report_t *report,  instance_div_t  *gpuInstances
  , instance_div_t  *instances
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
  double num_u32_ops = num_instances * numAd32OpsOfDivInst<uint32_t>(m);
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
  CUDA_CHECK(cudaMemcpy(instances, gpuInstances, sizeof(instance_div_t)*num_instances, cudaMemcpyDeviceToHost));

  // verifyResultsDiv(num_instances, instances);
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
  double num_u32_ops = num_instances * numAd32OpsOfDivInst<uint32_t>(m);
  double gigaopsu32 = num_u32_ops / (runtime_microsecs * 1000);

  printf( "CGBN quotient (num-instances = %d, num-word-len = %d, total-size: %d), \
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
  verifyResultsQuo(num_instances, instances);
}

void runGCD ( const uint32_t num_instances, const uint32_t cuda_block
  , cgbn_error_report_t *report,  instance_t  *gpuInstances
  , instance_t  *instances
) {

  const uint32_t ipb = cuda_block/TPI;

  unsigned long int elapsed = 0;
  struct timeval t_start, t_end, t_diff;
  gettimeofday(&t_start, NULL);

  for(int i = 0; i < GPU_RUNS_GCD; i++)
    kernel_gcd<<<(num_instances+ipb-1)/ipb, cuda_block>>>(report, gpuInstances, num_instances);

  cudaDeviceSynchronize();

  gettimeofday(&t_end, NULL);
  timeval_subtract(&t_diff, &t_end, &t_start);
  elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS_GCD;

  gpuAssert( cudaPeekAtLastError() );

  const uint32_t m = BITS / 32;
  double runtime_microsecs = elapsed;
  double num_u32_ops = num_instances * numAd32OpsOfGCDInst<uint32_t>(m);
  double gigaopsu32 = num_u32_ops / (runtime_microsecs * 1000);

  printf( "CGBN GCD (num-instances = %d, num-word-len = %d, total-size: %d), \
    averaged over %d runs: %lu microsecs, Gopsu32/sec: %.2f, Mil-Instances/sec: %.2f\n"
    , num_instances, m, num_instances * m, GPU_RUNS_GCD
    , elapsed, gigaopsu32, num_instances / runtime_microsecs
  );

  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);

  CUDA_CHECK(cudaMemcpy(instances, gpuInstances, sizeof(instance_t)*num_instances, cudaMemcpyDeviceToHost));

  verifyResultsGCD(num_instances, instances);
}


int main(int argc, char * argv[]) {
  if (argc != 2) {
      printf("Usage: %s <number-of-instances>\n", argv[0]);
      exit(1);
  }
      
  const int num_instances = atoi(argv[1]);

  instance_t          *instances, *gpuInstances;
  instance_div_t      *instancesDiv, *gpuInstancesDiv; 
	cgbn_error_report_t *report;

	CUDA_CHECK(cudaSetDevice(0));

  #if 0
  { // Add, Mul and Quo Benchmarks
	  instances = generate_instances(num_instances);
    CUDA_CHECK(cudaMalloc((void **)&gpuInstances, sizeof(instance_t)*num_instances));
    CUDA_CHECK(cudaMemcpy(gpuInstances, instances, sizeof(instance_t)*num_instances, cudaMemcpyHostToDevice));
    CUDA_CHECK(cgbn_error_report_alloc(&report)); 
    
    //runAdd (num_instances, 128, report, gpuInstances, instances);
    runMul (num_instances, 128, report, gpuInstances, instances);
    //// runPoly(num_instances, 128, report, gpuInstances, instances);
    //runQuo(num_instances, 128, report, gpuInstances, instances);

    free(instances);
    CUDA_CHECK(cudaFree(gpuInstances));
    CUDA_CHECK(cgbn_error_report_free(report));
  }
  #endif

  #if 1
  { // Division Benchmarks
    instancesDiv = generate_instances_div(num_instances);
    CUDA_CHECK(cudaMalloc((void **)&gpuInstancesDiv, sizeof(instance_div_t)*num_instances));
    CUDA_CHECK(cudaMemcpy(gpuInstancesDiv, instancesDiv, sizeof(instance_div_t)*num_instances, cudaMemcpyHostToDevice));
    CUDA_CHECK(cgbn_error_report_alloc(&report)); 

    runDiv(num_instances, 128, report, gpuInstancesDiv, instancesDiv);

    free(instancesDiv);
    CUDA_CHECK(cudaFree(gpuInstancesDiv));
    CUDA_CHECK(cgbn_error_report_free(report));
  }
  #endif

  #if 0
  { // GCD Benchmark
    instances = generate_instances_gcd(num_instances);
    CUDA_CHECK(cudaMalloc((void **)&gpuInstances, sizeof(instance_t)*num_instances));
    CUDA_CHECK(cudaMemcpy(gpuInstances, instances, sizeof(instance_t)*num_instances, cudaMemcpyHostToDevice));
    CUDA_CHECK(cgbn_error_report_alloc(&report)); 
    
    runGCD(num_instances, 128, report, gpuInstances, instances);

    free(instances);
    CUDA_CHECK(cudaFree(gpuInstances));
    CUDA_CHECK(cgbn_error_report_free(report));
  }
  #endif

  return 0;
}
