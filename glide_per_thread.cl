//This code is under MIT license. See license.txt for details.

//Calculate collatz glide by comparing parity vector(i.e. power of twos vs. power of threes)

#ifdef TEST
#define INIT_THREE_COUNT (SIEVE_STEP/4*3)
#define LUT_STEP 16
#endif

#define L_SHIFT (32 - LUT_STEP)
#define SIEVE_STEP 250
#define ALPHA 0.63092975357145743709952711434276 // == (log2 / log3)
#define VAL_LENGTH 16
#define LUT_IDX_MASK ((1u<<LUT_STEP)-1)

#ifndef MAX_COLLATZ_ITERATIONS
#define MAX_COLLATZ_ITERATIONS (1u<<20)
#endif

#define INIT_TWO_COUNT SIEVE_STEP
#define OVERFLOW_SHIFT 29
#define OVERSTEP_SHIFT 30

__constant uint power3[] = 
{
1u,3u,9u,27u,81u,243u,729u,2187u,6561u,19683u,
59049u,177147u,531441u,1594323u,4782969u,14348907u,43046721u,129140163u,387420489u,1162261467u,
3486784401u
};

/** @brief Create look-up table
    @param d_lut. s1 = Value to add in the 'jump'; s0 = Parity vector, presented as a bitfield. n-th bit = 
*/
__kernel void create_lut( __global uint2 * restrict d_lut)
{
  ulong addent = get_global_id(0);
  uint parityVec = 0u;
  #pragma unroll
  for(int i = 0; i < LUT_STEP; ++i){
    uint odd = convert_uint(addent) & 1u;
    addent = select(addent >> 1, rhadd(addent << 1, addent), convert_ulong(odd));
    parityVec |= odd << i; //Set i-th bit to odd(1) or even (0)
  }
  d_lut[get_global_id(0)] = (uint2)(parityVec, convert_uint(addent));
} 

/** @brief Create the standard of cumulative parity vector
    @param d_vecStandard Minimal requirement of cumulative parity vector
*/
__kernel void create_vec_standard(__global uint * restrict d_vecStandard)
{
  d_vecStandard[get_global_id(0)] = convert_uint_rtp(ALPHA * (get_global_id(0) + 1 + SIEVE_STEP));
}

ulong mul64L(uint a, uint b) 
{
  return upsample(mul_hi(a,b), a*b);
}

//a*b
uint2 mul64(uint a, uint b)
{
  return (uint2)(a*b, mul_hi(a,b));
}

//a + b
uint2 add64(uint2 a, uint b)
{
  a.lo += b;
  a.hi += a.lo < b;
  return a;
}

//a*b + c
uint2 mad64(uint a, uint b, uint c)
{
  return add64(mul64(a,b), c); 
}

//a*b + c + d
uint2 mad64_add(uint a, uint b, uint c, uint d)
{
  return add64(add64(mul64(a,b), c), d); 
}

//returns uint2(addent, p3).
//p3 = 0~LUT_STEP when no fail; (failStep + LUT_STEP) when failed.
uint2 check(uint2 lut, uint2 powerCount, __global const uint * restrict d_vecStandard)
{
  uint failStep = (uint)-1, oddCount = 0;
  #pragma unroll
  for(int i = 0; i < LUT_STEP; ++i){
    uint odd   = lut.s0 & (1u<<i);
    powerCount += (uint2)(1u, odd);
    oddCount   += odd; 
    uint fail  = select(powerCount.s0 + powerCount.s1, (uint)-1, powerCount.s1 >= d_vecStandard[powerCount.s0 - 1 - SIEVE_STEP]);
    failStep   = min(failStep, fail);
  }
  lut.s0 = select(failStep + LUT_STEP, oddCount, failStep == -1);
  return lut;
}

uint overflow(uint2 lut)
{
  return (lut.s1 != 0) << OVERFLOW_SHIFT ;
}

uint overstep(uint twoCount)
{
  return (twoCount >= (MAX_COLLATZ_ITERATIONS - LUT_STEP)) << OVERSTEP_SHIFT;
}

/** @brief Get maximal collatz glide steps

N0 = 2^250 * a + b0
After 250 collatz iterations : 
N250  = 3^x * a + b250, where a = ([initial offset] + [kernel offset]) + [thread offset] = offsetHi + gid
      = (3^x * (offsetHi) + b256 ) + 3^x * gid 
      = startVal + 3^x * gid

    @param d_maxStep Maximum collatz glide step of the thread
    @param d_maxPos Where the max step is ('a' above)
    @param d_totalSteps Accumulated collatz steps for scoring
    @param d_vecStandard Minimal requirement of threeCount of cumulative parity vector
    @param d_lut look-up table
    @param d_multiplier 3^x as above
    @param startVal startVal as above. Updated every time the kernel launches
    @param kno kernel launch 'serial number'. Updated every time the kernel launches
*/
__kernel void collatz_glide(
  __global uint4        * restrict d_results,
  __global const uint   * restrict d_vecStandard,
  __global const uint2  * restrict d_lut,
  __constant uint       * restrict d_multiplier,
  uint16 val, const uint kno )
{
  uint2 lut;
  {
    //initialize val = val + gid * multiplier
    lut = mad64(get_global_id(0), d_multiplier[0], val.s0);
    val.s0 = lut.s0;
    {
      lut = mad64_add(get_global_id(0), d_multiplier[1], val.s1, lut.s1);
      val.s1 = lut.s0;
      lut = mad64_add(get_global_id(0), d_multiplier[2], val.s2, lut.s1);
      val.s2 = lut.s0;
      lut = mad64_add(get_global_id(0), d_multiplier[3], val.s3, lut.s1);
      val.s3 = lut.s0;
      lut = mad64_add(get_global_id(0), d_multiplier[4], val.s4, lut.s1);
      val.s4 = lut.s0;
      lut = mad64_add(get_global_id(0), d_multiplier[5], val.s5, lut.s1);
      val.s5 = lut.s0;
      lut = mad64_add(get_global_id(0), d_multiplier[6], val.s6, lut.s1);
      val.s6 = lut.s0;
      lut = mad64_add(get_global_id(0), d_multiplier[7], val.s7, lut.s1);
      val.s7 = lut.s0;
      lut = mad64_add(get_global_id(0), d_multiplier[8], val.s8, lut.s1);
      val.s8 = lut.s0;
      lut = mad64_add(get_global_id(0), d_multiplier[9], val.s9, lut.s1);
      val.s9 = lut.s0;
      lut = mad64_add(get_global_id(0), d_multiplier[10], val.sa, lut.s1);
      val.sa = lut.s0;
      lut = mad64_add(get_global_id(0), d_multiplier[11], val.sb, lut.s1);
      val.sb = lut.s0;
      lut = mad64_add(get_global_id(0), d_multiplier[12], val.sc, lut.s1);
      val.sc = lut.s0;
      lut = mad64_add(get_global_id(0), d_multiplier[13], val.sd, lut.s1);
      val.sd = lut.s0;
      lut = mad64_add(get_global_id(0), d_multiplier[14], val.se, lut.s1);
      val.se = lut.s0;
      lut = mad64_add(get_global_id(0), d_multiplier[15], val.sf, lut.s1);
      val.sf = lut.s0;
    }
  
    uint2 powerCount = (uint2)(SIEVE_STEP, INIT_THREE_COUNT); //s0 : power of 2 so far. s1 : power of 3 so far
    lut = check(d_lut[val.s0 & LUT_IDX_MASK], powerCount, d_vecStandard);

    //Lookahead LUT_STEP steps if the value passes or fails.
    while(lut.s0 <= LUT_STEP){
      powerCount += (uint2)(LUT_STEP, lut.s0);
      // val = (val>>LUT_STEP) * mul + add
      uint mul = power3[lut.s0];
      {
        lut = mad64(mul, (val.s0 >> LUT_STEP) | (val.s1 << L_SHIFT), lut.s1);
        val.s0 = lut.s0;
        lut = mad64(mul, (val.s1 >> LUT_STEP) | (val.s2 << L_SHIFT), lut.s1);
        val.s1 = lut.s0;
        lut = mad64(mul, (val.s2 >> LUT_STEP) | (val.s3 << L_SHIFT), lut.s1);
        val.s2 = lut.s0;
        lut = mad64(mul, (val.s3 >> LUT_STEP) | (val.s4 << L_SHIFT), lut.s1);
        val.s3 = lut.s0;
        lut = mad64(mul, (val.s4 >> LUT_STEP) | (val.s5 << L_SHIFT), lut.s1);
        val.s4 = lut.s0;
        lut = mad64(mul, (val.s5 >> LUT_STEP) | (val.s6 << L_SHIFT), lut.s1);
        val.s5 = lut.s0;
        lut = mad64(mul, (val.s6 >> LUT_STEP) | (val.s7 << L_SHIFT), lut.s1);
        val.s6 = lut.s0;
        lut = mad64(mul, (val.s7 >> LUT_STEP) | (val.s8 << L_SHIFT), lut.s1);
        val.s7 = lut.s0;
        lut = mad64(mul, (val.s8 >> LUT_STEP) | (val.s9 << L_SHIFT), lut.s1);
        val.s8 = lut.s0;
        lut = mad64(mul, (val.s9 >> LUT_STEP) | (val.sa << L_SHIFT), lut.s1);
        val.s9 = lut.s0;
        lut = mad64(mul, (val.sa >> LUT_STEP) | (val.sb << L_SHIFT), lut.s1);
        val.sa = lut.s0;
        lut = mad64(mul, (val.sb >> LUT_STEP) | (val.sc << L_SHIFT), lut.s1);
        val.sb = lut.s0;
        lut = mad64(mul, (val.sc >> LUT_STEP) | (val.sd << L_SHIFT), lut.s1);
        val.sc = lut.s0;
        lut = mad64(mul, (val.sd >> LUT_STEP) | (val.se << L_SHIFT), lut.s1);
        val.sd = lut.s0;
        lut = mad64(mul, (val.se >> LUT_STEP) | (val.sf << L_SHIFT), lut.s1);
        val.se = lut.s0;
      }
      lut = mad64(mul, val.sf >> LUT_STEP, lut.s1);
      val.sf = lut.s0;
      mul = overflow(lut) | overstep(powerCount.s0);
      lut = check(d_lut[val.s0 & LUT_IDX_MASK], powerCount, d_vecStandard);
      lut.s0 |= mul;
    } //while(lut.s0 <= LUT_STEP)
  }
  lut.s0 -= LUT_STEP;
  //Compare and update the results
  val.s0123 = d_results[get_global_id(0)];
  val.s1 = select(val.s1, kno, lut.s0 > val.s0);
  val.s0 = max(val.s0, lut.s0);
  val.s23 = add64(val.s23, lut.s0);
  d_results[get_global_id(0)] = val.s0123;
}                          