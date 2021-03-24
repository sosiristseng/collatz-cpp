// This code is licensed under the MIT License (MIT). See LICENCE.txt for details

#pragma once

//Standard Libraries
#include <algorithm>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

//Boost libraries
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/algorithm/string.hpp>

#define BOOST_COMPUTE_HAVE_THREAD_LOCAL
#define BOOST_COMPUTE_THREAD_SAFE
#include <boost/compute.hpp>