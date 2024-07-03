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

namespace Realm {
namespace Upmem {
    extern Logger log_stream;

    DPUMRAMMemory::DPUMRAMMemory(Memory _me, DPU* _dpu, DPUStream* _stream, char* _base,
        size_t _size)
        : LocalManagedMemory(_me, _size, MKIND_MRAM, 512, Memory::DPU_MRAM_MEM, 0)
        , dpu(_dpu)
        , base(_base)
        , stream(_stream)
    {
    }

    DPUMRAMMemory::~DPUMRAMMemory(void) { }

    // these work, but they are SLOW
    void DPUMRAMMemory::get_bytes(off_t offset, void* dst, size_t size)
    {
        // use a blocking copy - host memory probably isn't pinned anyway
        {
            // we need to get the dpu_set_t stream
            CHECK_UPMEM(dpu_copy_from(*stream->get_stream(), "data", offset, dst, size));
        }
    }

    void DPUMRAMMemory::put_bytes(off_t offset, const void* src, size_t size)
    {
        // use a blocking copy - host memory probably isn't pinned anyway
        {
            // we need to get the dpu_set_t stream
            CHECK_UPMEM(dpu_broadcast_to(*stream->get_stream(), "data", offset, src, size,
                DPU_XFER_DEFAULT));
        }
    }

    void* DPUMRAMMemory::get_direct_ptr(off_t offset, size_t size)
    {
        return (void*)(base + offset);
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // class DPUMemcpy

    DPUMemcpy::DPUMemcpy(DPU* _dpu, DPUMemcpyKind _kind)
        : dpu(_dpu)
        , kind(_kind)
    {
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // class DPUMemcpyFence

    DPUMemcpyFence::DPUMemcpyFence(DPU* _dpu, DPUMemcpyKind _kind, DPUWorkFence* _fence)
        : DPUMemcpy(_dpu, _kind)
        , fence(_fence)
    {
        // log_stream.info() << "dpu memcpy fence " << this << " (fence = " << fence << ")
        // created";
    }

    void DPUMemcpyFence::execute(DPUStream* stream)
    {
        fence->enqueue_on_stream(stream);
#ifdef FORCE_DPU_STREAM_SYNCHRONIZE
        DPU_ASSERT(dpu_sync(*stream->get_stream()));
#endif
    }

}; // namespace Upmem
}; // namespace Realm