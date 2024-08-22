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
 
// This common header file is meant to be an "intermediate" between host and
// dpu device code.

/*
    https://github.com/CMU-SAFARI/prim-benchmarks/tree/main
    Juan Gómez-Luna, Izzat El Hajj, Ivan Fernandez, Christina Giannoula, Geraldo F.
   Oliveira, and Onur Mutlu, "Benchmarking Memory-centric Computing Systems: Analysis of
   Real Processing-in-Memory Hardware". 2021 12th International Green and Sustainable
   Computing Conference (IGSC). IEEE, 2021.
*/

#ifndef _UPMEM_COMMON_H_
#define _UPMEM_COMMON_H_

#ifdef DEVICE_DPU_CODE
#include <realm/upmem/legion_c_upmem.h>
#else 
#include <legion.h>
using namespace Legion;
#endif

#ifdef __cplusplus
extern "C" {

#include <stdint.h>

#endif
// Transfer size between MRAM and WRAM
#ifdef BL
#define BLOCK_SIZE_LOG2 BL
#define BLOCK_SIZE (1 << BLOCK_SIZE_LOG2)
#else
#define BLOCK_SIZE_LOG2 8
#define BLOCK_SIZE (1 << BLOCK_SIZE_LOG2)
#define BL BLOCK_SIZE_LOG2
#endif

// Data type
#ifdef UINT32
#define T uint32_t
#define DIV 2 // Shift right to divide by sizeof(T)
#elif UINT64
#define T uint64_t
#define DIV 3
#elif INT32
#define T int32_t
#define DIV 2
#elif INT64
#define T int64_t
#define DIV 3
#elif FLOAT
#define T float
#define DIV 2
#elif DOUBLE
#define T double
#define DIV 3
#elif CHAR
#define T char
#define DIV 0
#elif SHORT
#define T short
#define DIV 1
#else
#error Must define a valid type. See /realm/upmem/upmem_common.h
#endif

#define divceil(n, m) (((n)-1) / (m) + 1)
#define roundup(n, m) ((n / m) * m + m)

#ifdef __cplusplus
}
#endif

#endif