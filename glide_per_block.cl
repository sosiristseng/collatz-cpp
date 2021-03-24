//Calculates collatz glide step with 1200-step sieve, using 64*32 = 2048 bit integer



#define WG_SIZE 64u
#define VAL_LENGTH 64u
#define SIEVE_STEP 1200u
#define MAX_COLLATZ_ITERATIONS (1u<<16)
#define MAX_STEP 0xfffffffu
#define LUT_MASK ((1u<<LUT_STEP)-1u)

#ifdef TEST
#define LUT_STEP 15u
#define INIT_ODD_COUNT (SIEVE_STEP/4*3)
#endif

__constant uint power3[] = 
{
1u,3u,9u,27u,81u,243u,729u,2187u,6561u,19683u,
59049u,177147u,531441u,1594323u,4782969u,14348907u,43046721u,129140163u,387420489u,1162261467u,
3486784401u
};

/** 
  @brief Create look-up tables for collatz glide kernel

  CAUTION:
  1. Workgroup size must be WG_SIZE
  2. Total threads (global work size) must be 2^(LUT_STEP)
  
  @param d_cumuVec Inclusive prefix sum of collatz partity vector for the next N iterations. N == LUT_STEP, length == N*(2^N)
  @param d_addent Values to add in the 'jump'. length == 2^(LUT_STEP)
*/
__kernel void __attribute__ ((reqd_work_group_size(WG_SIZE, 1, 1)))
create_look_up_table(
  __global uchar *restrict d_cumuVec,
  __global uint  *restrict d_addent)
{
  const uint tid = get_local_id(0), gid = get_global_id(0), bid = get_group_id(0);
  ulong addent = gid;
  uint oddCnt = 0;
  const uint cacheIdx = tid * LUT_STEP;
  __local uchar l_cumuVec[WG_SIZE * LUT_STEP];
  #pragma unroll
  for(int i = 0; i < LUT_STEP; ++i){
    ulong odd = addent & 1ul;
    addent = select(addent>>1, rhadd(addent<<1, addent), odd);
    oddCnt += odd;
    l_cumuVec[i + cacheIdx] = oddCnt;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  event_t ev = async_work_group_copy(d_cumuVec + WG_SIZE * LUT_STEP * bid, l_cumuVec, WG_SIZE * LUT_STEP, 0);
  d_addent[gid] = addent;
  wait_group_events(1, &ev);
}

/** 
  @brief Calculate the propagate and generate bits with Kogge–Stone adder
  Reference : https://en.wikipedia.org/wiki/Kogge%E2%80%93Stone_adder
  @param l_propagate Propagate bits in the shared local memory
  @param l_generate Generate bits in the shared local memory
*/
void kogge_stone_adder(__local uint *restrict l_propagate, __local uint *restrict l_generate)
{
  const uint tid = get_local_id(0);
  #pragma unroll
  for(uint offset = 1; offset < WG_SIZE; offset *= 2){
    uint p = l_propagate[tid];
    uint g = l_generate[tid];
    if(tid >= offset){
      uint idx = tid - offset;
      g |= p & l_generate[idx];
      p &= l_propagate[idx];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    l_propagate[tid] = p;
    l_generate[tid] = g;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
} //kogge_stone_adder()

//Calculates l_a[] + l_b[]. Results are stored in l_a[]. Returns overflow bit
//Assuming the lengths of both l_a[] and l_b[] are (WG_SIZE+1)
uint addition(__local uint *restrict l_a, __local uint *restrict l_b)
{
  const uint tid = get_local_id(0);
  uint val = l_a[tid];
  uint carry = l_b[tid];
  uint overflow = l_a[WG_SIZE] | l_b[WG_SIZE];
  //calculates a+b, propagate and generate bits
  val += carry;
  carry = val < carry;
  barrier(CLK_LOCAL_MEM_FENCE);
  //Shifted by one element
  l_a[tid + 1] = (val == 0xffffffffu); //propagate bits
  l_b[tid + 1] = carry; //generate bits
  l_a[0] = 0;
  l_b[0] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  kogge_stone_adder(l_a + 1, l_b + 1);
  l_a[tid] = val + l_b[tid];
  barrier(CLK_LOCAL_MEM_FENCE);
  return overflow | l_b[WG_SIZE]; //overflow bit
}

ulong mul64L(uint a, uint b) 
{
  return upsample(mul_hi(a,b), a*b);
}

uint lut_idx(uint val)
{
  return val & LUT_MASK;
}

//WIP: calculate threadhold on-the-fly
//Returns which position would the value fail. Returns LUT_STEP when there's no fail (success)
uint fail_iter(
  __local uchar *restrict l_cumuVec,
  __local uint *restrict l_failIter,
  __global const uint *restrict d_oddThreshold,
  uint threeCount, uint nIters)
{
  const uint tid = get_local_id(0);
  if(tid < LUT_STEP){
    l_failIter[tid] = select((uint)LUT_STEP, tid, (l_cumuVec[tid] + threeCount) < d_oddThreshold[nIters + tid]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  //Reduction to get the failed step
  for(int offset = 16; offset >= 1; offset /= 2){
    uint pred = (tid < offset) && ((tid + offset) < LUT_STEP);
    if(pred){
      l_failIter[tid] = min(l_failIter[tid], l_failIter[tid + offset]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  return l_failIter[0];
}

event_t prefetch_lut(
  __local uchar *restrict l_cumuVec,
  __global const uchar *restrict d_cumuVec,
  __local uint *restrict l_addent,
  __global const uint *restrict d_addent,
  uint lutIdx,
  __global const uint *restrict d_oddThreshold,
  uint elapsedIter)
{
  prefetch(d_oddThreshold + elapsedIter, LUT_STEP);
  event_t ev = async_work_group_copy(l_cumuVec, d_cumuVec + lutIdx * LUT_STEP, LUT_STEP, 0);
  return async_work_group_copy(l_addent, d_addent + lutIdx, 1, ev);
}

uint is_overflow(uint overflowbits)
{
  return overflowbits != 0;
}

uint is_over_step(uint nIters)
{
  return nIters >= MAX_COLLATZ_ITERATIONS;
}

/** 
  @brief Get maximal collatz glide steps
  
  @par How this kernel works
  N0 = 2^10000 * k + b0, where 0 < b0 = survived residue < 2^10000
     = 2^10000 * (bid + kno) + b0
  After 10000 collatz iterations:
  N256 = 3^X * (bid + kno) + b10000, X == INIT_ODD_COUNT
  
  @param d_maxStep Maximum collatz glide step of the work group. Length == get_num_groups(0)
  @param d_maxK Where the max step is, in terms of k. Length == get_num_groups(0)
  @param d_totalSteps Accumulated collatz steps for scoring. Length == get_num_groups(0)
  @param d_cumuVec Cumulative odd # occurrence during a 'jump'. Size == N*2^N, N == LUT_STEP
  @param d_addent Values to add in the 'jump'. Size == 2^N, N == LUT_STEP
  @param d_oddThreshold Minimal divergent parity vector, in terms of inclusive prefix sum of power of 3's.
      If calculated power of 3's less than the threshold, drop the candidate. Counts from iteration (SIEVE_STEP+1).
  @param c_multiplier 3^X. Length == WG_SIZE
  @param c_baseVal Starting offset number after (SIEVE_STEP) collatz iterations. Length == WG_SIZE
  @param kno Kernel launch number updated every kernel launch. Could use y-axis instead of another argument.
*/ 
__kernel void __attribute__ ((reqd_work_group_size(WG_SIZE, 1, 1)))
collatz_glide(
  __global uint         *restrict d_maxStep,
  __global uint         *restrict d_maxK,
  __global ulong        *restrict d_totalSteps,
  __global const uchar  *restrict d_cumuVec,
  __global const uint   *restrict d_addent,
  __global const uint   *restrict d_oddThreshold,
  __constant uint       *restrict c_multiplier,
  __constant uint       *restrict c_baseVal,
  const int kno )
{
  //This kernel would take about 4KB of shared local memory per workgroup which contains 512 threads
  __local uint l_val[WG_SIZE + 1];
  __local uint l_carry[WG_SIZE + 1];
  __local uchar l_cumuVec[LUT_STEP];
  __local uint l_failIter[LUT_STEP];
  __local uint l_addent[1];
  
  //val = baseVal + multiplier * (bid + kno)
  const uint tid = get_local_id(0), bid = get_group_id(0);
  ulong addRes = c_baseVal[tid] + mul64L(c_multiplier[tid], bid + kno);
  l_val[tid] = addRes; //lower 32 bits
  l_carry[tid + 1] = addRes >> 32; //higher 32 bits
  l_carry[0] = 0;
  l_val[WG_SIZE] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  //Start prefetching look-up-table items before carry operation
  uint elapsedIter = 0, threeCount = INIT_ODD_COUNT;
  event_t ev = prefetch_lut(l_cumuVec, d_cumuVec, l_addent, d_addent, lut_idx(l_val[0]), d_oddThreshold, elapsedIter);
  addition(l_val, l_carry); //carry propagation
  wait_group_events(1, &ev); //Now we need look-up-table items
  //See if val fails
  uint pred = fail_iter(l_cumuVec, l_failIter, d_oddThreshold, threeCount, elapsedIter);
  while(pred == LUT_STEP){
    uint p3 = l_cumuVec[LUT_STEP-1], multiplier = power3[p3];
    //val = (val>>LUT_STEP) * multiplier + addent
    uint val = (l_val[tid] >> LUT_STEP) | (l_val[tid + 1] << (32-LUT_STEP));
    barrier(CLK_LOCAL_MEM_FENCE);
    l_val[tid] = val * multiplier; //lower 32 bits
    l_carry[tid + 1] = mul_hi(val, multiplier); //higher 32 bits
    l_carry[0] = l_addent[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    //prefetch next lut item in the middle
    ev = prefetch_lut(l_cumuVec, d_cumuVec, l_addent, d_addent, lut_idx(l_val[0] + l_addent[0]), d_oddThreshold, elapsedIter);
    uint overflow = addition(l_val, l_carry);
    elapsedIter += LUT_STEP;
    threeCount += p3;
    wait_group_events(1, &ev);
    //See if val fails
    pred = fail_iter(l_cumuVec, l_failIter, d_oddThreshold, threeCount, elapsedIter) | (is_overflow(overflow)<<16) | (is_over_step(elapsedIter)<<17);
  }
  //calculate steps and compare results to previous ones
  if(tid == 0){
    const uint failedStep = min((uint)LUT_STEP, pred + 1);
    uint step = select(elapsedIter + threeCount + failedStep + l_cumuVec[failedStep-1] + SIEVE_STEP, MAX_STEP, pred >> 16);
    uint stepOther = d_maxStep[bid];
    uint kOther    = d_maxK[bid];
    pred = step > stepOther;
    stepOther = select(stepOther, step, pred);
    kOther    = select(kOther, kno + bid, pred);
    d_maxStep[bid] = stepOther;
    d_maxK[bid] = kOther;
    d_totalSteps[bid] += step;
  }
}
