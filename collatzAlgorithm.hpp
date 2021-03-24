// This code is licensed under the MIT License (MIT). See LICENCE.txt for details
#pragma once
#include <cstdint>
#include <tuple>

namespace collatzOpenCL {

//One collatz iteration, returns tuple(result, odd)
template <typename IntTy>
inline auto collatz_op(IntTy val)
{
  auto odd = static_cast<uint32_t>(val & 1u); //Test the least significant bit
  val = odd ? (val + ((val + 1) >> 1)) : (val >> 1);
  return std::make_tuple(val, odd);
}

//Several collatz iterations, returns (result, oddCount)
template <typename IntTy>
inline auto collatz_op(IntTy val, const int nLoops)
{
  auto oddCount = uint32_t{ 0 };
  for (int i = 0; i < nLoops; ++i) {
    auto odd = uint32_t{ 0 };
    std::tie(val, odd) = collatz_op(val);
    oddCount += odd;
  }
  return std::make_tuple(val, oddCount);
}

template <typename T, typename U>
inline auto collatz_step(T val, U const& stoppingVal)
{
  auto stepCount = uint32_t{ 0 };
  while (val > stoppingVal) {
    auto odd = uint32_t{ 0 };
    std::tie(val, odd) = collatz_op(val);
    stepCount += 1 + odd;
  }
  return stepCount;
}

//full stopping time (delay steps)
template <typename T>
inline auto delay_step(T const& val)
{
  return collatz_step(val, 1);
}

//stopping time (glide steps)
template <typename T>
inline auto glide_step(T const& val)
{
  return (val <= 2) ? (0u) : (collatz_step(val, val - 1));
}

namespace detail {

template <typename T>
inline bool is_failed_sieve(T const& start, const uint32_t steps)
{
  auto val = start;
  for (unsigned int i = 0; i < steps; ++i) {
    std::tie(val, std::ignore) = collatz_op(val);
    if (val <= start) return true;
  }
  return false;
}

} //namespace detail

//Finds the 'survived' residue after collatz iterations.
template <typename T>
inline auto get_sieve_num(const uint32_t targetSieveStep, T startVal = 3)
{
  //searches the sieve number from n = 4k+3
  for (startVal |= 3; detail::is_failed_sieve(startVal, targetSieveStep); startVal += 4) { ; }
  return startVal;
}

} //namespace collatzOpenCL