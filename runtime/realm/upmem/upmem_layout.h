/* Copyright 2024 Stanford University, NVIDIA Corporation
 *                Los Alamos National Laboratory, Northwestern University
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

// Rank memory layout, shared by host and device code (plain C).
// Rank address v is interleaved over N DPUs in granules of G bytes:
//   DPU (v / G) % N, local offset (v / (N * G)) * G + v % G
// Local offsets are relative to REALM_UPMEM_MRAM_DATA_OFFSET in each DPU's MRAM.

#ifndef REALM_UPMEM_LAYOUT_H
#define REALM_UPMEM_LAYOUT_H

#include <stdint.h>

// MRAM below this offset is left to the kernel binary's __mram variables
#define REALM_UPMEM_MRAM_DATA_OFFSET (1u << 20)
#define REALM_UPMEM_MRAM_SIZE (64u << 20)
#define REALM_UPMEM_MRAM_DATA_SIZE (REALM_UPMEM_MRAM_SIZE - REALM_UPMEM_MRAM_DATA_OFFSET)

#define REALM_UPMEM_MAX_DPUS_PER_RANK 64

// rank memory pointers are rank addresses plus this (keeps offset 0 non-NULL)
#define REALM_UPMEM_RANK_PTR_BASE 8u

#ifdef __cplusplus
extern "C" {
#endif

// DPU (index within the rank) holding rank address v
static inline uint32_t realm_upmem_owner(uint64_t v, uint32_t nr_dpus, uint32_t granule)
{
  return (uint32_t)((v / granule) % nr_dpus);
}

// local MRAM data offset of rank address v on its owning DPU
static inline uint64_t realm_upmem_local(uint64_t v, uint32_t nr_dpus, uint32_t granule)
{
  return (v / ((uint64_t)granule * nr_dpus)) * granule + (v % granule);
}

// rank address of local offset l on DPU d (inverse of realm_upmem_local)
static inline uint64_t realm_upmem_rank_addr(uint64_t l, uint32_t d, uint32_t nr_dpus,
                                             uint32_t granule)
{
  return (l / granule) * ((uint64_t)granule * nr_dpus) + (uint64_t)d * granule +
         (l % granule);
}

// bytes of rank range [0, v) on DPU d, i.e. local start of DPU d's share of [v, ...)
static inline uint64_t realm_upmem_local_bound(uint64_t v, uint32_t d, uint32_t nr_dpus,
                                               uint32_t granule)
{
  uint64_t row = (uint64_t)granule * nr_dpus;
  uint64_t in_row = v % row;
  uint64_t lo = (uint64_t)d * granule;
  uint64_t part = 0;
  if(in_row > lo)
    part = ((in_row - lo) < granule) ? (in_row - lo) : granule;
  return (v / row) * granule + part;
}

#ifdef __cplusplus
}
#endif

#if defined(DEVICE_DPU_CODE)
#include <mram.h>

// set by Kernel::load
__host uint32_t REALM_UPMEM_DPU_ID;
__host uint32_t REALM_UPMEM_SHAPE[2]; // { DPUs in rank, granule }

// MRAM address of rank-memory local offset l on this DPU
#define REALM_UPMEM_MRAM(l)                                                              \
  ((__mram_ptr uint8_t *)(uintptr_t)(REALM_UPMEM_MRAM_DATA_OFFSET + (l)))

// rank address of a pointer into a rank memory (e.g. from an accessor)
#define REALM_UPMEM_RANK_ADDR(ptr) ((uint32_t)(uintptr_t)(ptr)-REALM_UPMEM_RANK_PTR_BASE)

// local MRAM address of a rank-memory pointer owned by this DPU
static inline __mram_ptr uint8_t *realm_upmem_local_ptr(const void *ptr)
{
  return REALM_UPMEM_MRAM(realm_upmem_local(REALM_UPMEM_RANK_ADDR(ptr),
                                            REALM_UPMEM_SHAPE[0], REALM_UPMEM_SHAPE[1]));
}

// this DPU's share of rank-memory bytes [ptr, ptr + bytes): contiguous in MRAM,
//  returns its start and sets *my_bytes
static inline __mram_ptr uint8_t *realm_upmem_my_share(const void *ptr, uint32_t bytes,
                                                       uint32_t *my_bytes)
{
  uint32_t v = REALM_UPMEM_RANK_ADDR(ptr);
  uint32_t lo = realm_upmem_local_bound(v, REALM_UPMEM_DPU_ID, REALM_UPMEM_SHAPE[0],
                                        REALM_UPMEM_SHAPE[1]);
  uint32_t hi = realm_upmem_local_bound(v + bytes, REALM_UPMEM_DPU_ID,
                                        REALM_UPMEM_SHAPE[0], REALM_UPMEM_SHAPE[1]);
  *my_bytes = hi - lo;
  return REALM_UPMEM_MRAM(lo);
}
#endif

#endif
