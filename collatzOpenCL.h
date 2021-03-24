// This code is licensed under the MIT License (MIT). See LICENCE.txt for details

#pragma once
#include "clHelper.h"
#include <iostream>

namespace collatzOpenCL {
//CONSTANTS
constexpr auto kernelFileName = "collatz.cl";
constexpr auto clearKernelName = "clearRes";
constexpr auto collatzKernelName = "collatz";
constexpr auto sieveStep = 256;
constexpr auto valLength = 16;
constexpr auto numQueues = 2; //Concurrent opencl command queues

void run_kernel_impl(const cl::Device & clDevice, std::istream & configFile, std::ostream & logger, std::ostream& output);

inline void run_kernel(const cl::Device& device, std::istream& configFile, std::ostream& output, std::ostream& logger)
{
  clh::ocl_error_handler(logger, run_kernel_impl, device, configFile, logger, output);
}
} //namespace collatzOpenCL