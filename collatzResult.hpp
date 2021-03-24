// This code is licensed under the MIT License (MIT). See LICENCE.txt for details
#pragma once

#include <algorithm>
#include <cstdint>
#include <exception>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>

#include <boost/lexical_cast.hpp>

namespace collatzOpenCL
{

class ValidateErr : public std::exception {
public:
  ValidateErr(const std::string& val, unsigned int expectedStep, unsigned int calculatedStep)
  {
    std::stringstream ss("Value :");
    ss << val << "\nExpected Step: " << expectedStep << "\nCalculated Step: " << calculatedStep << '\n';
    errMsg_.assign(ss.str());
  }
  const char* what() const override { return errMsg_.c_str(); }
private:
  std::string errMsg_;
};

template<typename MaxStepTy = uint32_t, typename MaxPosTy = uint64_t, typename TotalStepsTy = uint64_t>
struct Result {
  MaxStepTy maxStep = 0;
  MaxPosTy maxPos = 0;
  TotalStepsTy totalSteps = 0;
};

template<typename T, typename U, typename V>
inline decltype(auto) operator << (std::ostream& os, const Result<T,U,V>& result)
{
  return os << result.maxStep << '\n' << result.maxPos << '\n' << result.totalSteps;
}
template<typename T, typename U, typename V>
inline decltype(auto) operator >> (std::istream& is, Result<T,U,V>& result)
{
  return is >> result.maxStep >> result.maxPos >> result.totalSteps;
}

template<typename T, typename U, typename V, typename ValGen, typename CollatzFunc> 
inline auto compare_and_validate(const Result<T,U,V>& oldRes, const Result<T,U,V>& newRes, ValGen&& get_val, CollatzFunc&& collatz_step)
{
  //Only validate the results when new champion arrives
  if (newRes.maxStep > oldRes.maxStep) {
    auto val = get_val(newRes); //synthesize the value from the results
    auto expectedStep = collatz_step(val);
    if (newRes.maxStep != expectedStep) {
      throw ValidateErr{ boost::lexical_cast<std::string>(val), expectedStep, newRes.maxStep };
    }
    return Result<T, U, V>{ newRes.maxStep, newRes.maxPos, newRes.totalSteps + oldRes.totalSteps };
  } else {
    return Result<T, U, V>{ oldRes.maxStep, oldRes.maxPos, newRes.totalSteps + oldRes.totalSteps };
  }
}

template <typename MaxStepVec, typename MaxPosVec, typename TotalStepsVec> inline
auto gather_results(MaxStepVec&& maxStep, MaxPosVec&& maxPos, TotalStepsVec&& totalSteps)
{
  using std::cbegin; using std::cend; using std::next;
  auto pos = std::distance(cbegin(maxStep), std::max_element(cbegin(maxStep), cend(maxStep)));
  return Result<decltype(maxStep[0]), decltype(maxPos[0]), decltype(totalSteps[0])> {
    *next(cbegin(maxStep), pos),
    *next(cbegin(maxPos), pos),
    std::accumulate(cbegin(totalSteps), cend(totalSteps), 0)
  };
}

}