// This code is licensed under the MIT License (MIT). See LICENCE.txt for details
#pragma once

#include <boost/algorithm/string.hpp>
#include <fstream>
#include <map>

namespace collatzOpenCL {

inline auto read_config(const char* filename)
{
  auto ifs = std::ifstream{ filename };
  auto settings = std::map<std::string, std::string>{};
  for (std::string key, value; std::getline(ifs, key, '=') && std::getline(ifs, value, '\n'); /*empty*/) {
    settings.emplace(boost::trim_copy(key), boost::trim_copy(value));
  }
  return settings;
}

}
