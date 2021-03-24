#include "sieve.h"

#include <tuple>
#include <cmath>
#include <algorithm>

#include "collatzGeneric.h"

const uint32_t power3[] =
{
  1u, 3u, 9u, 27u, 81u, 243u, 729u, 2187u, 6561u, 19683u,
  59049u, 177147u, 531441u, 1594323u, 4782969u, 14348907u, 43046721u, 129140163u, 387420489u, 1162261467u, 3486784401u
};

auto create_sieve(const uint32_t k) 
{
  const auto scoreBoard[2] = { -1.0, scoreOdd };
  const uint32_t halfStep = k >> 1;
  std::vector <float> negNadir(1u << halfStep);
  std::vector <std::tuple<uint32_t/*sieveVal*/, uint32_t/*power3*/, uint32_t/*bCoefficient*/> > halfSieve;
  //generate look-up tables with half the step, so we can jump to target sieve with 2-step approach.
  for (uint32_t i = 0; i < negNadir.size(); ++i) {
    uint32_t powerOf3 = 0, b = i;
    float score = 0.0f, nadir = 100.0f;
    for (uint32_t j = 0; j < halfStep; ++j) {
      auto odd = collatz_op(b);
      score += scoreBoard[odd];
      powerOf3 += odd;
      nadir = std::min(nadir, score);
    }
    negNadir[i] = -nadir;
    //if it survived the halfsieve, add it to look-up table
    if (nadir > 0.0f) {
      halfSieve.emplace_back(i, powerOf3, b);
    }
  }
  const uint32_t MASK = (1u << halfStep) - 1u;
  std::vector<uint32_t> resultSieve;
  //outer loop iterates upper half bits; inner loop : lower half
  for (uint32_t startHi = 0; startHi < negNadir.size(); ++startHi) {
    for (uint32_t i = 0; i < halfSieve.size(); ++i) {
      uint32_t sieveVal, powerOf3, b;
      std::tie(sieveVal, powerOf3, b) = halfSieve[i];
      float score = powerOf3 * scoreOdd - halfStep + powerOf3;
      const uint32_t transformed = startHi * power3[powerOf3] + b; //transformed value after 'halfStep' collatz iterations
      if (score > negNadir[transformed & MASK]) {
        resultSieve.push_back((startHi << halfStep) + sieveVal);
      }
    }
  }
  return resultSieve;
} 