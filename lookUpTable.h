#pragma once

#include "clHelper.h"
#include <cstdint>

namespace collatzOpenCL
{
cl::vector<cl_ulong> getLUT(const uint32_t step);

cl::vector<cl_uint> getDelayTable(const uint32_t step);
} //namespace collatzOpenCL