//This code is under the MIT license. Copyright (c) 2016 Wen-Wei Tseng. All rights reserved. 
//this kernel requires WG_SIZE_LOG2 constant

#ifdef TEST_COMPILE
#define WG_SIZE_LOG2 8
#endif //#ifdef TEST_COMPILE

//Adapted from two-stage reduction from AMD website: 
//http://developer.amd.com/resources/documentation-articles/articles-whitepapers/opencl-optimization-case-study-simple-reductions/
__kernel __attribute__((reqd_work_group_size(1u<<WG_SIZE_LOG2, 1, 1)))
void reduce(
  __global uint  * restrict g_maxStepOut,
  __global ulong * restrict g_maxPosOut, 
  __global ulong * restrict g_totalStepsOut,
  __global uint  * restrict g_maxStepIn, 
  __global ulong * restrict g_maxPosIn, 
  __global ulong * restrict g_totalStepsIn,
  const    uint             inputLength
){
  __local uint  l_maxStep   [1u<<WG_SIZE_LOG2];
  __local uint  l_maxId     [1u<<WG_SIZE_LOG2];
  __local ulong l_totalSteps[1u<<WG_SIZE_LOG2];
  uint id          = get_global_id(0);
  uint maxIdx      = id;
  uint maxStep     = g_maxStepIn[id];
  ulong totalSteps = g_totalStepsIn[id];
  //serially iterate over input arrays, with coalesced global memory reads
  for(id += get_global_size(0); id < inputLength; id += get_global_size(0)){
    uint maxStepOther = g_maxStepIn[id];
    maxIdx            = select(id, maxIdx, maxStepOther > maxStep);
    maxStep           = max(maxStep, maxStepOther);
    totalSteps       += g_totalStepsIn[id];
  } //for(i)
  id               = get_local_id(0);
  l_maxStep[id]    = maxStep;
  l_maxId[id]      = maxIdx;
  l_totalSteps[id] = totalSteps;
  //parallel reduction
  #pragma unroll WG_SIZE_LOG2
  for(uint offset = 1u<<(WG_SIZE_LOG2-1); offset > 0; offset >>=1 ) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if(id < offset) 
    {
      maxStep           = l_maxStep[id];
      maxIdx            = l_maxId[id];
      uint idOther      = id + offset; //position in shared local memory
      uint maxStepOther = l_maxStep[idOther];
      uint maxIdxOther  = l_maxIdx[idOther];
      l_maxIdx[id]      = select(maxIdx, maxIdxOther, maxStepOther > maxStep);
      l_maxStep[id]     = max(maxStep, maxStepOther);
      l_totalSteps[id] += l_totalSteps[idOther];
    } //if()
  } //for()
  if(id == 0) {
    g_maxStepOut   [get_group_id(0)]  = l_maxStep[0];
    g_maxPosOut    [get_group_id(0)]  = g_maxPosIn[l_maxIdx[0]];
    g_totalStepsOut[get_group_id(0)]  = l_totalSteps[0];
  }
} //reduce()