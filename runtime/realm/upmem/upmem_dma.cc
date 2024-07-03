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

#include "realm/upmem/upmem_dma.h"

namespace Realm {

  extern Logger log_xd;
  extern Logger log_taskreg;

  namespace Upmem {
    extern Logger log_dpu;
    extern Logger log_stream;
    extern Logger log_dpudma;    

    ////////////////////////////////////////////////////////////////////////
    //
    // class DPUfillChannel

      DPUfillChannel::DPUfillChannel(DPU *_dpu, BackgroundWorkManager *bgwork)
        : SingleXDQChannel<DPUfillChannel,DPUfillXferDes>(bgwork,
                                                          XFER_DPU_IN_MRAM,
                                                          stringbuilder() << "upmem fill channel (dpu=" << _dpu->info->index << ")")
        , dpu(_dpu)
      {
        std::vector<Memory> local_dpu_mems;
        local_dpu_mems.push_back(dpu->fbmem->me);

        // look for any other local memories that belong to our context
        const Node& n = get_runtime()->nodes[Network::my_node_id];
        for(std::vector<MemoryImpl *>::const_iterator it = n.memories.begin();
            it != n.memories.end();
            ++it) {
          UpmemDeviceMemoryInfo *cdm = (*it)->find_module_specific<UpmemDeviceMemoryInfo>();
          if(!cdm) continue;
          if(cdm->device_id != dpu->device_id) continue;
          local_dpu_mems.push_back((*it)->me);
        }

        unsigned bw = 300000;  // HACK - estimate at 300 GB/s
        unsigned latency = 250;  // HACK - estimate at 250 ns
        unsigned frag_overhead = 2000;  // HACK - estimate at 2 us

        add_path(Memory::NO_MEMORY, local_dpu_mems,
                 bw, latency, frag_overhead, XFER_DPU_IN_FB)
          .set_max_dim(2);

        xdq.add_to_manager(bgwork);
      }

      XferDes *DPUfillChannel::create_xfer_des(uintptr_t dma_op,
                                               NodeID launch_node,
                                               XferDesID guid,
                                               const std::vector<XferDesPortInfo>& inputs_info,
                                               const std::vector<XferDesPortInfo>& outputs_info,
                                               int priority,
                                               XferDesRedopInfo redop_info,
                                               const void *fill_data,
                                               size_t fill_size,
                                               size_t fill_total)
      {
        assert(redop_info.id == 0);
        return new DPUfillXferDes(dma_op, this, launch_node, guid,
                                  inputs_info, outputs_info,
                                  priority,
                                  fill_data, fill_size, fill_total);
      }

      long DPUfillChannel::submit(Request** requests, long nr)
      {
        // unused
        assert(0);
        return 0;
      }

  }; // namespace Upmem
};   // namespace Realm