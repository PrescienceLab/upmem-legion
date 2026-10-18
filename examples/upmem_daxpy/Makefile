# Copyright 2024 Stanford University, Los Alamos National Laboratory,
#                Northwestern University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

ifndef LG_RT_DIR
$(error LG_RT_DIR variable is not defined, aborting build)
endif

TYPE=INT32
NR_TASKLETS ?= 16

DEBUG           ?= 0
MAX_DIM         ?= 3
OUTPUT_LEVEL    ?= LEVEL_INFO
USE_CUDA        ?= 0
USE_HIP         ?= 0
USE_GASNET      ?= 0
USE_HDF         ?= 0
ALT_MAPPERS     ?= 0
USE_UPMEM       ?= 1

OUTFILE		?= upmem_daxpy
GEN_SRC		?= host/daxpy.cc
GEN_GPU_SRC	?=
GEN_UPMEM_SRC	?= dpu/daxpy_kernel.cc

INC_FLAGS	?= -Iinclude
CC_FLAGS	?= -D$(TYPE) -DLEGION_MAX_NUM_PROCS=256
NVCC_FLAGS	?=
HIPCC_FLAGS	?=
UPMEMCC_FLAGS	?= -D$(TYPE) -DNR_TASKLETS=$(NR_TASKLETS)
GASNET_FLAGS	?=
LD_FLAGS	?=

include $(LG_RT_DIR)/runtime.mk
