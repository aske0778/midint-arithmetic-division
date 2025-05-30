#ifndef CGBN_KERNELS
#define CGBN_KERNELS

// Declare the instance type
typedef struct {
  cgbn_mem_t<BITS> a;
  cgbn_mem_t<BITS> b;
  cgbn_mem_t<BITS> sum;
} instance_t;

// Declare the division instance type
typedef struct {
  cgbn_mem_t<BITS> a;
  cgbn_mem_t<BITS> b;
  cgbn_mem_t<BITS> quo;
  cgbn_mem_t<BITS> rem;
} instance_div_t;


// helpful typedefs for the kernel
typedef cgbn_context_t<TPI>         context_t;
typedef cgbn_env_t<context_t, BITS> env_t;

/***********************/
/*** Addition Kernel ***/
/***********************/

__global__ void kernel_add(cgbn_error_report_t *report, instance_t *instances, uint32_t count) {
  int32_t instance;
  
  // decode an instance number from the blockIdx and threadIdx
  instance=(blockIdx.x*blockDim.x + threadIdx.x)/TPI;
  if(instance>=count)
    return;

  //context_t      bn_context(cgbn_report_monitor/*, report, instance*/);   // construct a context
  context_t      bn_context(cgbn_no_checks, NULL, instance);
  env_t          bn_env(bn_context.env<env_t>());                     // construct an environment for 1024-bit math
  env_t::cgbn_t  a, b, r;                                             // define a, b, r as 1024-bit bignums

  cgbn_load(bn_env, a, &(instances[instance].a));      // load my instance's a value
  cgbn_load(bn_env, b, &(instances[instance].b));      // load my instance's b value
  cgbn_add(bn_env, r, a, b);                           // r=a+b
  cgbn_store(bn_env, &(instances[instance].sum), r);   // store r into sum
}

__global__ void kernel_6adds(cgbn_error_report_t *report, instance_t *instances, uint32_t count) {
  int32_t instance;
  
  // decode an instance number from the blockIdx and threadIdx
  instance=(blockIdx.x*blockDim.x + threadIdx.x)/TPI;
  if(instance>=count)
    return;

  context_t      bn_context(cgbn_no_checks, NULL, instance);
  env_t          bn_env(bn_context.env<env_t>());                     // construct an environment for 1024-bit math
  env_t::cgbn_t  a, b, t1, t2;                                        // define a, b, r as 1024-bit bignums

  cgbn_load(bn_env, a, &(instances[instance].a));      // load my instance's a value
  cgbn_load(bn_env, b, &(instances[instance].b));      // load my instance's b value
  
  cgbn_add(bn_env, t1, a,   b);                           // t1=a+b
  cgbn_add(bn_env, t2, t1, t1);                           // t2=t1+t1
  cgbn_add(bn_env, t1, t2,  b);                           // t1=t2+b
  cgbn_add(bn_env, t2, t1, t1);                           // t2=t1+t1
  cgbn_add(bn_env,  a, t2, t1);                           // a=t2+t1
  cgbn_add(bn_env, t1,  a,  b);                           // t1=a+b
  
  cgbn_store(bn_env, &(instances[instance].sum), t1);     // store t1 into sum
}


/***********************/
/*** Multiply Kernel ***/
/***********************/

__global__ void kernel_mul(cgbn_error_report_t *report, instance_t *instances, uint32_t count) {
  int32_t instance;
  
  // decode an instance number from the blockIdx and threadIdx
  instance=(blockIdx.x*blockDim.x + threadIdx.x)/TPI;
  if(instance>=count)
    return;

  //context_t      bn_context(cgbn_report_monitor/*, report, instance*/);   // construct a context
  context_t      bn_context(cgbn_no_checks, NULL, instance);
  env_t          bn_env(bn_context.env<env_t>());                     // construct an environment for 1024-bit math
  env_t::cgbn_t  a, b, r;                                             // define a, b, r as 1024-bit bignums

  cgbn_load(bn_env, a, &(instances[instance].a));      // load my instance's a value
  cgbn_load(bn_env, b, &(instances[instance].b));      // load my instance's b value
  cgbn_mul(bn_env, r, a, b);                           // r=a+b
  cgbn_store(bn_env, &(instances[instance].sum), r);   // store r into sum
}

__global__ void kernel_poly(cgbn_error_report_t *report, instance_t *instances, uint32_t count) {
  int32_t instance;
  
  // decode an instance number from the blockIdx and threadIdx
  instance=(blockIdx.x*blockDim.x + threadIdx.x)/TPI;
  if(instance>=count)
    return;

  //context_t      bn_context(cgbn_report_monitor/*, report, instance*/);   // construct a context
  context_t      bn_context(cgbn_no_checks, NULL, instance);
  env_t          bn_env(bn_context.env<env_t>());                     // construct an environment for 1024-bit math
  env_t::cgbn_t  a, b, a2, a2pb, b2, b2pb, prod, ab, r;                   // define a, b, r as 1024-bit bignums

  cgbn_load(bn_env, a, &(instances[instance].a));      // load my instance's a value
  cgbn_load(bn_env, b, &(instances[instance].b));      // load my instance's b value
  cgbn_mul(bn_env, a2, a, a);                           // a2=a*a
  cgbn_add(bn_env, a2pb, a2, b);
  cgbn_mul(bn_env, b2, b, b);                           // b2=b*b
  cgbn_add(bn_env, b2pb, b2, b);
  // prod = bmul a2pb b2pb
  cgbn_mul(bn_env, prod, a2pb, b2pb);
  // ab   = bmul a  b
  cgbn_mul(bn_env, ab, a, b);
  // badd prod ab
  cgbn_add(bn_env, r, prod, ab);
  cgbn_store(bn_env, &(instances[instance].sum), r);   // store r into sum
}

/************************/
/*** Division Kernels ***/
/************************/

__global__ void kernel_quo(cgbn_error_report_t *report, instance_t *instances, uint32_t count) {
  int32_t instance;
  
  instance=(blockIdx.x*blockDim.x + threadIdx.x)/TPI;
  if(instance>=count)
    return;

  context_t      bn_context(cgbn_no_checks, NULL, instance);
  env_t          bn_env(bn_context.env<env_t>());
  env_t::cgbn_t  a, b, r;                              

  cgbn_load(bn_env, a, &(instances[instance].a)); 
  cgbn_load(bn_env, b, &(instances[instance].b));    
  cgbn_div(bn_env, r, a, b);                       
  cgbn_store(bn_env, &(instances[instance].sum), r); 
}


__global__ void kernel_div(cgbn_error_report_t *report, instance_div_t *instances, uint32_t count) {
  int32_t instance;
  
  instance=(blockIdx.x*blockDim.x + threadIdx.x)/TPI;
  if(instance>=count)
    return;

  context_t      bn_context(cgbn_no_checks, NULL, instance);
  env_t          bn_env(bn_context.env<env_t>());      
  env_t::cgbn_t  a, b, q, r;

  cgbn_load(bn_env, a, &(instances[instance].a));  
  cgbn_load(bn_env, b, &(instances[instance].b));  
  cgbn_div_rem(bn_env, q, r, a, b);           
  cgbn_store(bn_env, &(instances[instance].quo), q);
  cgbn_store(bn_env, &(instances[instance].rem), r);
}

/******************/
/*** GCD Kernel ***/
/******************/

__global__ void kernel_gcd(cgbn_error_report_t *report, instance_t *instances, uint32_t count) {
  int32_t instance;
  
  instance=(blockIdx.x*blockDim.x + threadIdx.x)/TPI;
  if(instance>=count)
    return;

  context_t      bn_context(cgbn_no_checks, NULL, instance);
  env_t          bn_env(bn_context.env<env_t>());                
  env_t::cgbn_t  a, b, r;                                        

  cgbn_load(bn_env, a, &(instances[instance].a));   
  cgbn_load(bn_env, b, &(instances[instance].b));      
  cgbn_gcd(bn_env, r, a, b);                         
  cgbn_store(bn_env, &(instances[instance].sum), r);  
}

#endif // CGBN_KERNELS
