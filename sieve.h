#pragma once

#include <vector>
#include <cstdint>

//creates a list of surviving residues under 2^k, k must be even
auto create_sieve(const uint32_t k);
