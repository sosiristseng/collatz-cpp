#ifdef TEST
//Constants required while compiling this kernel
#define LUTSIZE_LOG2 16     //Size of look-up table, in log2(size)
#define DELAYSIZE_LOG2 20   //Size of delay table, in log2(size)
#define INITSTEP 350        //collatz steps added in the precalculation
#endif //#ifdef TEST_COMPILE

#define VAL_LENGTH 16 //15 uints for value itself and 1 for overflow

#define LUTMASK ((1u<<LUTSIZE_LOG2)-1)
#define DELAYMASK ((1u<<DELAYSIZE_LOG2)-1)
#define MAXSTEP 0xFFFFFFFu

__constant uint power3[] = 
{
1u,3u,9u,27u,81u,243u,729u,2187u,6561u,19683u,
59049u,177147u,531441u,1594323u,4782969u,14348907u,43046721u,129140163u,387420489u,1162261467u,
3486784401u
};

inline ulong mul64L(uint a, uint b) 
{
  return upsample(mul_hi(a,b), a*b);
}

#ifdef VARIABLE_LENGTH //Functions for variable length kernel

//Assuming at least one element is not zero
inline uint getValLength(uint* val)
{
  uint idx = VAL_LENGTH - 1;
  while(val[idx] == 0) --idx;
  return idx + 1;
}
inline uint isOverflow(uint valLength, uint* val)
{
  return valLength >= VAL_LENGTH;
}
inline uint isNormalExit(uint valLength, uint* val)
{
  return valLength == 1 && val[0] <= DELAYMASK;
}

#else //Functions for fixed length kernel

inline uint isOverflow(uint valLength, uint* val)
{
  return val[VAL_LENGTH - 1] > 0;
}
inline uint isNormalExit(uint valLength, uint* val)
{
  uint pred = 0;
  #pragma unroll
  for(uint i = 1; i < VAL_LENGTH-1; ++i) {
    pred |= val[i];
  }
  return pred == 0 && val[0] <= DELAYMASK;
}
#endif //#ifdef VARIABLE_LENGTH

inline uint isOverSteps(uint stepCount)
{
  return stepCount >= MAXSTEP;
}

//////////////////////////////////////////////////////////////////////
// 
// Collatz openCL kernel to find the counter-example of Collatz conjecture, optimized by the 256-step sieve
// 
// N0 = 2**256 * k + b0, where 0 <= b0 = sieveNum < 2**256
//    = 2**256 * (gid + kOffset) + b0
// After 256 collatz iterations:
// N256 = 3**X * (gid + kOffset) + b256
//      = (3**X * gid) + (3**X * kOffset + b256)
//
/////////////////////////////////////////////////////////////////////////

__kernel void collatz(
  __global uint        * restrict g_maxStep,    /* maximum collatz step */
  __global ulong       * restrict g_maxPos,     /* the very kOffset where the max step is */
  __global ulong       * restrict g_totalSteps, /* total collatz steps calculated */
  __global const ulong * restrict g_lut,        /* look-up table. lo: powerOf3 ; hi: addent */
  __global const uint  * restrict g_delays,     /* collatz steps for # < 2**(DELAYSIZE_LOG2) */
  __constant uint      * restrict c_multiplier, /* 3**X */
  const uint16 c_addent,                        /* 3**X * kOffset + b256, should be updated when launching kernels  */
  const ulong kOffset
){
  uint val[VAL_LENGTH];  
  vstore16(c_addent, 0, val); //val = c_addent
  uint pred = get_global_id(0); //pred as multiplier
  //val += gid * c_multiplier
  ulong addRes = val[0] + mul64L(pred, c_multiplier[0]);
  val[0] = convert_uint(addRes);
  #pragma unroll
  for(uint i = 1; i < VAL_LENGTH; ++i) {
    addRes = (addRes>>32) + val[i] + mul64L(pred, c_multiplier[i]);
    val[i] = convert_uint(addRes);
  } //for()
#ifdef VARIABLE_LENGTH
  uint valLength = getValLength(val);
#else
#define valLength (VAL_LENGTH-1)
#endif // #ifdef VARIABLE_LENGTH
  uint stepCount = INITSTEP;
  do {
    addRes     = g_lut[val[0] & LUTMASK]; //most time-consuming global mem. access in this kernel
    pred       = convert_uint(addRes);
    stepCount += pred + LUTSIZE_LOG2;
    pred       = power3[pred]; //pred as multiplier
    //val = (val >> LUTSIZE_LOG2) * multiplier + addend, only "valLength" numbers in val array are calculated
    for(uint i = 0; i < valLength; ++i) 
    {
      addRes = (addRes >> 32) + mul64L(pred, (val[i] >> LUTSIZE_LOG2) | (val[i+1] << (32 - LUTSIZE_LOG2)));
      val[i] = convert_uint(addRes);
    } //for()
    val[valLength] = convert_uint(addRes >> 32);
#ifdef VARIABLE_LENGTH
    //valLength changes by 1 at most 
    valLength += val[valLength] > 0;
    valLength -= val[valLength] == 0 && val[valLength - 1] == 0;
#endif //#ifdef VARIABLE_LENGTH
    pred = (isOverflow(valLength, val) << 2) | (isOverSteps(stepCount) << 1) | isNormalExit(valLength, val);
  } while(pred == 0);
  stepCount                      += g_delays[val[0] & DELAYMASK];
  stepCount                       = select(stepCount, MAXSTEP, pred > 1);
  pred                            = g_maxStep[get_global_id(0)];
  addRes                          = g_maxPos[get_global_id(0)];
  g_totalSteps[get_global_id(0)] += stepCount;
  g_maxStep[get_global_id(0)]     = max(stepCount, pred);
  g_maxPos[get_global_id(0)]      = select(addRes, kOffset + get_global_id(0), convert_ulong(stepCount > pred));
} //collatzVariableLength()

//clearRes() : clears result buffers, could use clEnqueueFillBuffer() in openCL 1.2 or above
__kernel void clearRes( 
  __global uint * restrict g_maxStep, /* maximum step for this thread */
  __global ulong * restrict g_maxPos, /*position where the max step is*/
  __global ulong * restrict g_totalSteps /* total collatz (delay) steps calculated */
) {
  g_maxStep[get_global_id(0)]    = 0u;
  g_maxPos[get_global_id(0)]     = 0ul;
  g_totalSteps[get_global_id(0)] = 0ul;
} //clearRes()