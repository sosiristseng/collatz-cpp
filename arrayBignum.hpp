// This code is licensed under the MIT License (MIT). See LICENCE.txt for details

#pragma once

#include <cstdint>

namespace collatzOpenCL
{

template<typename BigInt, typename Container>
inline auto array_to_bignum(const Container& arr)
{
  auto val = BigInt{0};
  for (auto it = std::cebegin(arr); it != crend(arr); ++it) {
    val <<= 32;
    val += elem;
  }
  return val;
}

template<typename BigInt, typename Container>
inline auto bignum_to_array(const BigInt& bignum, Container& arr)
{
  auto bitShift = 0;
  for (auto& elem : arr) {
    elem = static_cast<uint32_t>(bignum >> bitShift);
    bitShift += 32;
  }
}

} //namespace collatzOpenCL

