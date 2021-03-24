// This code is licensed under the MIT License (MIT). See LICENCE.txt for details

#include "collatzOpenCL.h"
#include "arrayBignum.h"
#include "collatzConfig.h"
#include "validateErr.h"
#include "collatzGeneric.h"
#include "collatzResult.h"
#include "fileToStr.h"
#include "lookUpTable.h"
#include <algorithm>
#include <boost/multiprecision/cpp_int.hpp>
#include <iostream>

namespace collatzOpenCL {
namespace mp = boost::multiprecision;
using BigInt = boost::multiprecision::uint512_t;

struct Numbers {
  BigInt addent;
  uint32_t powerOf3;
  BigInt multiplier;
  Numbers(const Config& settings):
    addent(settings.startLo),
    powerOf3(collatz_op(addent, sieveStep)),
    multiplier(boost::multiprecision::pow(BigInt{3}, powerOf3))
  {}
}; //struct Numbers

std::string make_build_options_from_settings(const Config& settings, const Numbers& nums)
{
  return settings.buildOptions + clh::define_constant("LUTSIZE_LOG2", settings.lutSizeLog2) + clh::define_constant("DELAYSIZE_LOG2", settings.delaySizeLog2) + clh::define_constant("INITSTEP", nums.powerOf3 + sieveStep);
}
cl_uint16 make_uint16(const BigInt& val)
{
  cl_uint16 res;
  bignum_to_array(val, res.s);
  return res;
}
cl::vector<cl_uint> make_arr(const BigInt& val)
{
  cl::vector<cl_uint> res(valLength, 0);
  bignum_to_array(val, res);
  return res;
}

struct InputBuffers {
  cl::Buffer dLUT;
  cl::Buffer dDelays;
  cl::Buffer dMultiplier;

  InputBuffers(const cl::Context& clContext, const Numbers& nums, const Config& settings)
  {
    dLUT = clh::alloc(clContext, getLUT(settings.lutSizeLog2), true);
    dDelays = clh::alloc(clContext, getDelayTable(settings.delaySizeLog2), true);
    dMultiplier = clh::alloc(clContext, make_arr(nums.multiplier), true);
  }
}; //Inputbuffers

struct CollatzQueue {
  clh::Queue queue;
  clh::Ranges ranges;
  cl::Kernel clearKernel;
  cl::Kernel collatzKernel;
  cl::vector<cl_uint> hMaxSteps;
  cl::vector<cl_ulong> hMaxPos;
  cl::vector<cl_ulong> hTotalSteps;
  cl::Buffer dMaxSteps;
  cl::Buffer dMaxPos;
  cl::Buffer dTotalSteps;

  CollatzQueue(const cl::Program& program, const Config& settings, const InputBuffers& inBuf) :
    queue{ cl::CommandQueue(program.getInfo<CL_PROGRAM_CONTEXT>()) },
    ranges{ settings.globalWorkSize, settings.localWorkSize },
    clearKernel{ program, clearKernelName },
    collatzKernel{ program, collatzKernelName },
    hMaxSteps(settings.globalWorkSize, 0),
    hMaxPos(settings.globalWorkSize, 0),
    hTotalSteps(settings.globalWorkSize, 0),
    dMaxSteps{ clh::alloc(queue.context(), clh::get_byte_size(hMaxSteps)) },
    dMaxPos{ clh::alloc(queue.context(), clh::get_byte_size(hMaxPos)) },
    dTotalSteps{ clh::alloc(queue.context(), clh::get_byte_size(hMaxPos)) }
  {
    clh::set_args(clearKernel, dMaxSteps, dMaxPos, dTotalSteps);
    clh::set_args(collatzKernel, dMaxSteps, dMaxPos, dTotalSteps, inBuf.dLUT, inBuf.dDelays, inBuf.dMultiplier);
  }

  void queue_kernels(const Config& settings, const Numbers& nums, cl_ulong& kOffset)
  {
    constexpr auto indexOfAddent = 6;
    constexpr auto indexOfkOffset = 7;
    queue.run(clearKernel, ranges);
    for (auto i = settings.kernelsPerReduction; i > 0; --i, kOffset += settings.globalWorkSize) {
      collatzKernel.setArg(indexOfAddent, make_uint16(nums.addent + nums.multiplier * kOffset));
      collatzKernel.setArg(indexOfkOffset, kOffset);
      queue.run(collatzKernel, ranges);
    }
  }

  Result read_result()
  {
    queue.copy(dMaxSteps, hMaxSteps, CL_FALSE);
    queue.copy(dMaxPos, hMaxPos, CL_FALSE);
    queue.copy(dTotalSteps, hTotalSteps, CL_TRUE);
    return gather_results(hMaxSteps, hMaxPos, hTotalSteps);
  }
}; //struct CollatzQueue

Result worker_loop(cl::vector<CollatzQueue>& queues, const Config& settings, const Numbers& nums)
{
  Result result;
  cl_ulong kOffset = settings.startHi;
  for (auto& q : queues) {
    q.queue_kernels(settings, nums, kOffset);
  }
  do {
    for (auto& q : queues) {
      result = compare_and_validate(result, q.read_result(), BigInt{ settings.startLo });
      q.queue_kernels(settings, nums, kOffset);
    }
  } while (kOffset < settings.range);
  for (auto& q : queues) {
    result = compare_and_validate(result, q.read_result(), BigInt{ settings.startLo });
  }
  return result;
}

void run_kernel_impl(const cl::Device & clDevice, std::istream & configFile, std::ostream & logger, std::ostream& output)
{
  cl::Context clContext(clDevice);
  Config settings(configFile);
  Numbers nums(settings);
  InputBuffers inputBuffers(clContext, nums, settings);
  auto program = clh::build_program(clContext, file_to_str(kernelFileName), make_build_options_from_settings(settings, nums));
  cl::vector<CollatzQueue> collatzQueues
  (
    numQueues,
    CollatzQueue
    (
      program,
      settings,
      inputBuffers
    )
  );
  output << worker_loop(collatzQueues, settings, nums);
}

} //namespace collatzOpenCL