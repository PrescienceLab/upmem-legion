/* Copyright 2024 Stanford University, Los Alamos National Laboratory,
 *                Northwestern University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

extern "C" {
#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <stdint.h>
}

#include <realm/upmem/legion_c_upmem.h>
#include <common.h>

#define BLOCK_BYTES 512

typedef struct __DPU_LAUNCH_ARGS {
  char padding[256];
} __attribute__((aligned(8))) __DPU_LAUNCH_ARGS;

__host __DPU_LAUNCH_ARGS ARGS;
// OP_SUM (sum of acc_x) result for this DPU
__host int64_t PARTIAL;

BARRIER_INIT(sum_barrier, NR_TASKLETS);
int64_t tasklet_sums[NR_TASKLETS];

__dma_aligned TYPE buf_x[NR_TASKLETS][BLOCK_BYTES / sizeof(TYPE)];
__dma_aligned TYPE buf_y[NR_TASKLETS][BLOCK_BYTES / sizeof(TYPE)];
__dma_aligned TYPE buf_z[NR_TASKLETS][BLOCK_BYTES / sizeof(TYPE)];

static int sum_x(const DPU_LAUNCH_ARGS *args, unsigned int t)
{
  uint32_t nx;
  __mram_ptr uint8_t *x = realm_upmem_my_share(args->acc_x.ptr(args->rect.lo),
                                               args->rect.volume() * sizeof(TYPE), &nx);
  int64_t sum = 0;
  for(uint32_t off = t * BLOCK_BYTES; off < nx; off += NR_TASKLETS * BLOCK_BYTES) {
    uint32_t len = (nx - off < BLOCK_BYTES) ? (nx - off) : BLOCK_BYTES;
    mram_read(x + off, buf_x[t], (len + 7) & ~7u);
    for(uint32_t i = 0; i < len / sizeof(TYPE); i++)
      sum += buf_x[t][i];
  }
  tasklet_sums[t] = sum;
  barrier_wait(&sum_barrier);
  if(t == 0) {
    for(unsigned int i = 1; i < NR_TASKLETS; i++)
      sum += tasklet_sums[i];
    PARTIAL = sum;
  }
  return 0;
}

int main(void)
{
  const DPU_LAUNCH_ARGS *args = (const DPU_LAUNCH_ARGS *)(&ARGS);
  unsigned int t = me();
  if(args->op == OP_SUM)
    return sum_x(args, t);

  // this DPU's slice of each field (identical in size, the mapper row-aligns fields)
  uint32_t bytes = args->rect.volume() * sizeof(TYPE);
  uint32_t nx, ny, nz;
  __mram_ptr uint8_t *x = realm_upmem_my_share(args->acc_x.ptr(args->rect.lo), bytes, &nx);
  __mram_ptr uint8_t *y = realm_upmem_my_share(args->acc_y.ptr(args->rect.lo), bytes, &ny);
  __mram_ptr uint8_t *z = realm_upmem_my_share(args->acc_z.ptr(args->rect.lo), bytes, &nz);
  assert(nx == ny && ny == nz);

  for(uint32_t off = t * BLOCK_BYTES; off < nx; off += NR_TASKLETS * BLOCK_BYTES) {
    uint32_t len = (nx - off < BLOCK_BYTES) ? (nx - off) : BLOCK_BYTES;
    uint32_t xlen = (len + 7) & ~7u; // MRAM DMA moves whole 8-byte words
    mram_read(x + off, buf_x[t], xlen);
    mram_read(y + off, buf_y[t], xlen);
    if(xlen != len)
      mram_read(z + off, buf_z[t], xlen); // keep the bytes past the end
    for(uint32_t i = 0; i < len / sizeof(TYPE); i++)
      buf_z[t][i] = args->alpha * buf_x[t][i] + buf_y[t][i];
    mram_write(buf_z[t], z + off, xlen);
  }
  return 0;
}
