// This code is licensed under the MIT License (MIT). See LICENCE.txt for details
#pragma once

#include <string>
#include <fstream>
#include <sstream>

inline auto file_to_str(const char* fileName)
{
  std::ifstream ifs{ fileName };
  std::stringstream buffer;
  buffer << ifs.rdbuf();
  return buffer.str();
}