
// This common header file is meant to be an "intermediate" between host and 
// dpu device code. 

/*
    https://github.com/CMU-SAFARI/prim-benchmarks/tree/main
    Juan Gómez-Luna, Izzat El Hajj, Ivan Fernandez, Christina Giannoula, Geraldo F. Oliveira, and Onur Mutlu, 
    "Benchmarking Memory-centric Computing Systems: Analysis of Real Processing-in-Memory Hardware". 
    2021 12th International Green and Sustainable Computing Conference (IGSC). IEEE, 2021.
*/

#ifndef _UPMEM_COMMON_H_
#define _UPMEM_COMMON_H_

#ifdef __cplusplus
extern "C" {

#include <stdint.h>

#endif 

// Structures used by both the host and the dpu to communicate information
typedef struct {
    uint32_t size;
    uint32_t transfer_size;
} dpu_arguments_t;

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
#endif

#define divceil(n, m) (((n)-1) / (m) + 1)
#define roundup(n, m) ((n / m) * m + m)

#ifdef __cplusplus
}
#endif 

#endif