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

#ifndef REALM_UPMEM_MEMORY_H
#define REALM_UPMEM_MEMORY_H

enum DPUMemcpyKind {
    DPU_MEMCPY_HOST_TO_DEVICE,
    DPU_MEMCPY_DEVICE_TO_HOST,
    DPU_MEMCPY_DEVICE_TO_DEVICE,
};

#include "realm/mem_impl.h"

// #include "realm/upmem/upmem_internal.h"

#include "realm/upmem/upmem_stream.h"
#include "realm/upmem/upmem_workers.h"

namespace Realm {
namespace Upmem {
    // forward declaration
    // internal.h
    class DPU;
    // stream.h
    class DPUStream;
    // workers.h
    class DPUWorkFence;

    class DPUMRAMMemory : public LocalManagedMemory {
    public:
        DPUMRAMMemory(Memory _me, DPU* _dpu, DPUStream* _stream, char* _base, size_t _size);

        virtual ~DPUMRAMMemory(void);

        // these work, but they are SLOW
        virtual void get_bytes(off_t offset, void* dst, size_t size);
        virtual void put_bytes(off_t offset, const void* src, size_t size);

        virtual void* get_direct_ptr(off_t offset, size_t size);

    public:
        DPU* dpu;
        char* base;
        DPUStream* stream;
    }; // end class DPUMRAMMemory

    // An abstract base class for all DPU memcpy operations
    class DPUMemcpy {
    public:
        DPUMemcpy(DPU* _dpu, DPUMemcpyKind _kind);
        virtual ~DPUMemcpy(void) { }

    public:
        virtual void execute(DPUStream* stream) = 0;

    public:
        DPU* const dpu;

    protected:
        DPUMemcpyKind kind;
    }; // end class DPUMemcpy

    class DPUMemcpyFence : public DPUMemcpy {
    public:
        DPUMemcpyFence(DPU* _dpu, DPUMemcpyKind _kind, DPUWorkFence* _fence);

        virtual void execute(DPUStream* stream);

    protected:
        DPUWorkFence* fence;
    }; // end class DPUMemcpyFence

}; // namespace Upmem
}; // namespace Realm

#endif
