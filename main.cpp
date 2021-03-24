//Compare 64-bit vs 32-bit int in lut generation

#include <cstdint>
#include <iostream>

using namespace std;

auto collatz_op = [](auto val)
{
	auto odd = val & 1u;
	return (odd == 1) ? (val + ((val + 1) >> 1)) : (val >> 1);
};

constexpr auto lutStep = 20;
constexpr auto numElems = 1 << lutStep;

int main()
{
	for (auto i = 0; i < numElems; ++i) {
		//cout << "i=" << i << '\n';
		uint32_t i32 = i;
		uint64_t i64 = i;
		for (auto j = 0; j < lutStep; ++j) {
			i32 = collatz_op(i32);
			i64 = collatz_op(i64);
			if (static_cast<uint64_t>(i32) != i64) {
				cout << "Difference detected in i=" << i << ", j=" << j << '\n';
				cout << "i32=" << i32 << ", i64=" << i64 << '\n';
				return 1;
			}
		}
	}
	return 0;
}