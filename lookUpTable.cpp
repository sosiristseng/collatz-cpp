///////////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2016 Wen-Wei Tseng. All rights reserved.
//
// This code is licensed under the MIT License (MIT).
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
///////////////////////////////////////////////////////////////////////////////

#include "lookUpTable.h"
#include "collatzGeneric.h"
namespace collatzOpenCL
{
//const cl_uint power3[] = {
//  1u, 3u, 9u, 27u, 81u, 243u, 729u, 2187u, 6561u, 19683u,
//  59049u, 177147u, 531441u, 1594323u, 4782969u, 14348907u, 43046721u, 129140163u, 387420489u, 1162261467u, 3486784401u
//}; //power3[]

cl_uint getDelay(cl::vector<cl_uint>& table, const cl_ulong index)
{
  constexpr cl_uint UNDEFINED = 0xffffffffu;
  if (table[index] == UNDEFINED) {
    cl_ulong i = index;
    cl_uint stepCount = 0;
    do {
      stepCount += 1 + collatz_op(i);
    } while (i >= table.size());
    table[index] = stepCount + getDelay(table, i);
  }
  return table[index];
} //getDelay()

cl::vector<cl_ulong> getLUT(const uint32_t step)
{
  cl::vector<cl_ulong> result(1ull << step);
  for (uint32_t i = 0; i < result.size(); ++i) {
    cl_ulong b = i;
    auto p3 = collatz_op(b, step);
    result[i] = (b << 32) | p3;
  }
  return result;
} //getLUT()

cl::vector<cl_uint> getDelayTable(const uint32_t step)
{
  constexpr cl_uint UNDEFINED = 0xffffffffu;
  cl::vector<cl_uint> delayTable(1ull << step, UNDEFINED);
  delayTable[0] = delayTable[1] = 0;
  for (auto i = delayTable.size() - 1; i > 1; --i) {
    getDelay(delayTable, i);
  }
  return delayTable;
} //getDelayTable()
} //namespace collatzOpenCL