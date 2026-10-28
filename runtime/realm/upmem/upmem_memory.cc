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

#include "realm/upmem/upmem_memory.h"
#include "realm/upmem/upmem_internal.h"
#include "realm/inst_impl.h"

namespace Realm {
  namespace Upmem {
    extern Logger log_stream;

    DPUZCMemory::DPUZCMemory(Memory _me, char *_base, size_t _size)
      : LocalManagedMemory(_me, _size, MKIND_ZEROCOPY, 512, Memory::Z_COPY_MEM, 0)
      , base(_base)
    {}

    DPUZCMemory::~DPUZCMemory(void) {}

    // these work, but they are SLOW
    void DPUZCMemory::get_bytes(off_t offset, void *dst, size_t size) { assert(0); }

    void DPUZCMemory::put_bytes(off_t offset, const void *src, size_t size) { assert(0); }

    void *DPUZCMemory::get_direct_ptr(off_t offset, size_t size)
    {
      return (void *)(base + offset);
    }

    DPUMRAMMemory::DPUMRAMMemory(Memory _me, DPU *_dpu, DPUStream *_stream, char *_base,
                                 size_t _size)
      : LocalManagedMemory(_me, _size, MKIND_MRAM, 512, Memory::DPU_MRAM_MEM, 0)
      , dpu(_dpu)
      , base(_base)
      , stream(_stream)
    {}

    DPUMRAMMemory::~DPUMRAMMemory(void) {}

    void DPUMRAMMemory::get_bytes(off_t offset, void *dst, size_t size)
    {
      dpu->copy_sync(DPU_XFER_FROM_DPU, dst, offset, size);
    }

    void DPUMRAMMemory::put_bytes(off_t offset, const void *src, size_t size)
    {
      dpu->copy_sync(DPU_XFER_TO_DPU, const_cast<void *>(src), offset, size);
    }

    void *DPUMRAMMemory::get_direct_ptr(off_t offset, size_t size)
    {
      // not dereferenceable on the host - a rank address plus REALM_UPMEM_RANK_PTR_BASE
      return (void *)(base + offset);
    }

    void DPUMRAMMemory::round_to_rows(RegionInstanceImpl *inst)
    {
      InstanceLayoutGeneric *layout = inst->metadata.layout;
      if(!dpu->blocked || !layout || inst->metadata.ext_resource)
        return;
      const size_t row = dpu->row_bytes();
      layout->bytes_used = (layout->bytes_used + row - 1) / row * row;
      if((layout->alignment_reqd == 0) || ((row % layout->alignment_reqd) == 0))
        layout->alignment_reqd = row;
      else
        layout->alignment_reqd *= row;
    }

    MemoryImpl::AllocationResult
    DPUMRAMMemory::allocate_storage_deferrable(RegionInstanceImpl *inst,
                                               bool need_alloc_result, Event precondition)
    {
      round_to_rows(inst);
      AllocationResult result =
          LocalManagedMemory::allocate_storage_deferrable(inst, need_alloc_result,
                                                          precondition);
      if(dpu->blocked && (result == ALLOC_INSTANT_SUCCESS))
        dpu->note_block(inst);
      return result;
    }

    MemoryImpl::AllocationResult
    DPUMRAMMemory::allocate_storage_immediate(RegionInstanceImpl *inst,
                                              bool need_alloc_result, bool poisoned,
                                              TimeLimit work_until)
    {
      round_to_rows(inst);
      AllocationResult result = LocalManagedMemory::allocate_storage_immediate(
          inst, need_alloc_result, poisoned, work_until);
      if(dpu->blocked && (result == ALLOC_INSTANT_SUCCESS))
        dpu->note_block(inst);
      return result;
    }

    MemoryImpl::AllocationResult
    DPUMRAMMemory::reuse_allocated_range(RegionInstanceImpl *old_inst,
                                         std::vector<RegionInstanceImpl *> &new_insts)
    {
      for(RegionInstanceImpl *inst : new_insts)
        round_to_rows(inst);
      // new offsets are only known once notified - copies look them up lazily
      return LocalManagedMemory::reuse_allocated_range(old_inst, new_insts);
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // class DPUMemcpy

    DPUMemcpy::DPUMemcpy(DPU *_dpu, DPUMemcpyKind _kind)
      : dpu(_dpu)
      , kind(_kind)
    {}

    ////////////////////////////////////////////////////////////////////////
    //
    // class DPUMemcpyFence

    DPUMemcpyFence::DPUMemcpyFence(DPU *_dpu, DPUMemcpyKind _kind, DPUWorkFence *_fence)
      : DPUMemcpy(_dpu, _kind)
      , fence(_fence)
    {
      // log_stream.info() << "dpu memcpy fence " << this << " (fence = " << fence << ")
      // created";
    }

    void DPUMemcpyFence::execute(DPUStream *stream)
    {
      fence->enqueue_on_stream(stream);
#ifdef FORCE_DPU_STREAM_SYNCHRONIZE
      CHECK_UPMEM(dpu_sync(*stream->get_stream()));
#endif
    }

  }; // namespace Upmem
};   // namespace Realm