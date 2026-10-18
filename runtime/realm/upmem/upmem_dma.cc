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
    // class DPUXferDes

    static DPU *mem_to_dpu(const MemoryImpl *mem)
    {
      if(ID(mem->me).is_memory()) {
        const DPUMRAMMemory *mram = dynamic_cast<const DPUMRAMMemory *>(mem);
        if(mram)
          return mram->dpu;

        // see if it has UpmemDeviceMemoryInfo with a valid dpu
        const UpmemDeviceMemoryInfo *cdm =
            mem->find_module_specific<UpmemDeviceMemoryInfo>();
        if(cdm && cdm->dpu)
          return cdm->dpu;       
      }
      // not a dpu-associated memory
      return 0;
    }

    DPUXferDes::DPUXferDes(uintptr_t _dma_op, Channel *_channel, NodeID _launch_node,
                           XferDesID _guid,
                           const std::vector<XferDesPortInfo> &inputs_info,
                           const std::vector<XferDesPortInfo> &outputs_info,
                           int _priority)
      : XferDes(_dma_op, _channel, _launch_node, _guid, inputs_info, outputs_info,
                _priority, 0, 0)
    {
      kind = XFER_DPU_IN_MRAM; // TODO: is this needed at all?

      src_dpus.resize(inputs_info.size(), 0);
      for(size_t i = 0; i < input_ports.size(); i++) {
        src_dpus[i] = mem_to_dpu(input_ports[i].mem);
        // sanity-check
        if(input_ports[i].mem->kind == MemoryImpl::MKIND_MRAM)
          assert(src_dpus[i]);
      }

      dst_dpus.resize(outputs_info.size(), 0);
      dst_is_ipc.resize(outputs_info.size(), false);
      for(size_t i = 0; i < output_ports.size(); i++) {
        dst_dpus[i] = mem_to_dpu(output_ports[i].mem);
        if(output_ports[i].mem->kind == MemoryImpl::MKIND_MRAM) {
          // sanity-check
          assert(dst_dpus[i]);
        } else {
          // assume a memory owned by another node is ipc
          if(NodeID(ID(output_ports[i].mem->me).memory_owner_node()) !=
             Network::my_node_id)
            dst_is_ipc[i] = true;
        }
      }
    }

    long DPUXferDes::get_requests(Request **requests, long nr)
    {
      // unused
      assert(0);
      return 0;
    }

    bool DPUXferDes::progress_xd(DPUChannel *channel, TimeLimit work_until)
    {
      bool did_work = false;

      ReadSequenceCache rseqcache(this, MEGABYTE);
      WriteSequenceCache wseqcache(this, MEGABYTE);

      while(true) {
        size_t min_xfer_size = 4096; // TODO: make controllable
        size_t max_bytes = get_addresses(min_xfer_size, &rseqcache);
        if(max_bytes == 0) {
          break;
        }

        XferPort *in_port = 0, *out_port = 0;
        size_t in_span_start = 0, out_span_start = 0;
        DPU *in_dpu = 0, *out_dpu = 0;
        if(input_control.current_io_port >= 0) {
          in_port = &input_ports[input_control.current_io_port];
          in_span_start = in_port->local_bytes_total;
          in_dpu = src_dpus[input_control.current_io_port];
        }
        if(output_control.current_io_port >= 0) {
          out_port = &output_ports[output_control.current_io_port];
          out_span_start = out_port->local_bytes_total;
          out_dpu = dst_dpus[output_control.current_io_port];
        }

        size_t total_bytes = 0;
        if(in_port != 0) {
          if(out_port != 0) {
            // input and output both exist - transfer what we can
            log_xd.info() << "upmem memcpy chunk: min=" << min_xfer_size
                          << " max=" << max_bytes;

            uintptr_t in_base =
                reinterpret_cast<uintptr_t>(in_port->mem->get_direct_ptr(0, 0));
            uintptr_t out_base =
                reinterpret_cast<uintptr_t>(out_port->mem->get_direct_ptr(0, 0));

            // exactly one side is a rank memory (no d2d/p2p paths are registered)
            assert((in_dpu != 0) != (out_dpu != 0));
            DPU *dpu = in_dpu ? in_dpu : out_dpu;
            DPUMemcpyKind copy_type = in_dpu ? DPU_XFER_FROM_DPU : DPU_XFER_TO_DPU;

            while(total_bytes < max_bytes) {
              AddressListCursor &in_alc = in_port->addrcursor;
              AddressListCursor &out_alc = out_port->addrcursor;

              uintptr_t in_offset = in_alc.get_offset();
              uintptr_t out_offset = out_alc.get_offset();

              // strided copies are issued one line at a time - the rank layout
              //  turns any line into several DPU pushes anyway
              size_t bytes = std::min(std::min(in_alc.remaining(0), out_alc.remaining(0)),
                                      max_bytes - total_bytes);

              // rank memory offsets are rank addresses
              void *host;
              uint64_t rank_addr;
              if(in_dpu) {
                rank_addr = in_offset;
                host = reinterpret_cast<void *>(out_base + out_offset);
              } else {
                rank_addr = out_offset;
                host = reinterpret_cast<void *>(in_base + in_offset);
              }

              dpu->enqueue_copy(copy_type, host, rank_addr, bytes);

              log_dpudma.info() << "dpu memcpy: rank=" << dpu->device_id
                                << " rank_addr=" << std::hex << rank_addr
                                << " host=" << host << std::dec << " bytes=" << bytes
                                << " kind=" << (in_dpu ? "d2h" : "h2d");

              in_alc.advance(0, bytes);
              out_alc.advance(0, bytes);
              total_bytes += bytes;

              // stop if it's been too long, but make sure we do at least the
              //  minimum number of bytes
              if((total_bytes >= min_xfer_size) && work_until.is_expired())
                break;
            }

            if(total_bytes > 0) {
              add_reference(); // released by transfer completion
              log_dpudma.info() << "dpu memcpy fence: rank=" << dpu->device_id
                                << " xd=" << std::hex << guid << std::dec
                                << " bytes=" << total_bytes;
              dpu->stream->add_notification(new DPUTransferCompletion(
                  this, input_control.current_io_port, in_span_start, total_bytes,
                  output_control.current_io_port, out_span_start, total_bytes));
            }
          } else { // out_port == 0
            // input but no output, so skip input bytes
            total_bytes = max_bytes;
            in_port->addrcursor.skip_bytes(total_bytes);

            rseqcache.add_span(input_control.current_io_port, in_span_start, total_bytes);
          }
        } else { // in_port == 0
          if(out_port != 0) {
            // output but no input, so skip output bytes
            total_bytes = max_bytes;
            out_port->addrcursor.skip_bytes(total_bytes);
          } else { // out_port == 0
            // skipping both input and output is possible for simultaneous
            //  gather+scatter
            total_bytes = max_bytes;

            wseqcache.add_span(output_control.current_io_port, out_span_start,
                               total_bytes);
          }
        }

        if(total_bytes > 0) {
          did_work = true;

          bool done = record_address_consumption(total_bytes, total_bytes);

          if(done || work_until.is_expired())
            break;
        }
      }

      rseqcache.flush();
      wseqcache.flush();

      return did_work;
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // class DPUChannel

    DPUChannel::DPUChannel(DPU *_src_dpu, XferDesKind _kind,
                           BackgroundWorkManager *bgwork)
      : SingleXDQChannel<DPUChannel, DPUXferDes>(bgwork, _kind, "help")
    // stringbuilder() << "upmem channel (dpu=" << _src_dpu->info->index
    //                 << " kind=" << (int)_kind << ")")

    {
      src_dpu = _src_dpu;

      // switch out of ordered mode if multi-threaded dma is requested
      if(_src_dpu->module->config->cfg_multithread_dma)
        xdq.ordered_mode = false;

      std::vector<Memory> local_dpu_mems;
      local_dpu_mems.push_back(src_dpu->mram->me);

      std::vector<Memory> peer_dpu_mems;
      peer_dpu_mems.insert(peer_dpu_mems.end(), src_dpu->peer_mram.begin(),
                           src_dpu->peer_mram.end());

      // ipc
      for(std::vector<DPU::UpmemIpcMapping>::const_iterator it =
              src_dpu->upmemipc_mappings.begin();
          it != src_dpu->upmemipc_mappings.end(); ++it)
      {
        peer_dpu_mems.push_back(it->mem);
      }

      // look for any other local memories that belong to our context or
      //  peer-able contexts
      const Node &n = get_runtime()->nodes[Network::my_node_id];
      for(std::vector<MemoryImpl *>::const_iterator it = n.memories.begin();
          it != n.memories.end(); ++it) {
        UpmemDeviceMemoryInfo *cdm = (*it)->find_module_specific<UpmemDeviceMemoryInfo>();
        if(!cdm)
          continue;
        
        if(cdm->device_id == src_dpu->device_id) {
          local_dpu_mems.push_back((*it)->me);
        } else {
          // if the other context is associated with a dpu and we've got peer
          //  access, use it
          // TODO: add option to enable peer access at this point?  might be
          //  expensive...
          if(cdm->dpu && (src_dpu->info->peers.count(cdm->dpu->info->device) > 0))
            peer_dpu_mems.push_back((*it)->me);
        }
      }

      std::vector<Memory> mapped_cpu_mems;
      mapped_cpu_mems.insert(mapped_cpu_mems.end(), src_dpu->pinned_sysmems.begin(),
                             src_dpu->pinned_sysmems.end());

      switch(_kind) {
      case XFER_DPU_TO_MRAM:
      {
        unsigned bw = 10000;           // HACK - estimate at 10 GB/s
        unsigned latency = 1000;       // HACK - estimate at 1 us
        unsigned frag_overhead = 2000; // HACK - estimate at 2 us

        add_path(mapped_cpu_mems, local_dpu_mems, bw, latency, frag_overhead,
                 XFER_DPU_TO_MRAM)
            .set_max_dim(2); // D->H cudamemcpy3d is unrolled into 2d copies

        break;
      }

      case XFER_DPU_FROM_MRAM:
      {
        unsigned bw = 10000;           // HACK - estimate at 10 GB/s
        unsigned latency = 1000;       // HACK - estimate at 1 us
        unsigned frag_overhead = 2000; // HACK - estimate at 2 us

        add_path(local_dpu_mems, mapped_cpu_mems, bw, latency, frag_overhead,
                 XFER_DPU_FROM_MRAM)
            .set_max_dim(2); // H->D cudamemcpy3d is unrolled into 2d copies

        break;
      }

      case XFER_DPU_IN_MRAM:
      {
        // self-path
        unsigned bw = 200000;          // HACK - estimate at 200 GB/s
        unsigned latency = 250;        // HACK - estimate at 250 ns
        unsigned frag_overhead = 2000; // HACK - estimate at 2 us

        add_path(local_dpu_mems, local_dpu_mems, bw, latency, frag_overhead,
                 XFER_DPU_IN_MRAM)
            .set_max_dim(3);

        break;
      }

      case XFER_DPU_PEER_MRAM:
      {
        // just do paths to peers - they'll do the other side
        unsigned bw = 50000;           // HACK - estimate at 50 GB/s
        unsigned latency = 1000;       // HACK - estimate at 1 us
        unsigned frag_overhead = 2000; // HACK - estimate at 2 us

        add_path(local_dpu_mems, peer_dpu_mems, bw, latency, frag_overhead,
                 XFER_DPU_PEER_MRAM)
            .set_max_dim(3);

        break;
      }

      default:
        assert(0);
      }
    }

    DPUChannel::~DPUChannel() {}

    XferDes *DPUChannel::create_xfer_des(uintptr_t dma_op, NodeID launch_node,
                                         XferDesID guid,
                                         const std::vector<XferDesPortInfo> &inputs_info,
                                         const std::vector<XferDesPortInfo> &outputs_info,
                                         int priority, XferDesRedopInfo redop_info,
                                         const void *fill_data, size_t fill_size,
                                         size_t fill_total)
    {
      assert(redop_info.id == 0);
      assert(fill_size == 0);
      return new DPUXferDes(dma_op, this, launch_node, guid, inputs_info, outputs_info,
                            priority);
    }

    long DPUChannel::submit(Request **requests, long nr)
    {
      // unused
      assert(0);
      return 0;
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // class DPUTransfercompletion

    DPUTransferCompletion::DPUTransferCompletion(XferDes *_xd, int _read_port_idx,
                                                 size_t _read_offset, size_t _read_size,
                                                 int _write_port_idx,
                                                 size_t _write_offset, size_t _write_size)
      : xd(_xd)
      , read_port_idx(_read_port_idx)
      , read_offset(_read_offset)
      , read_size(_read_size)
      , write_port_idx(_write_port_idx)
      , write_offset(_write_offset)
      , write_size(_write_size)
    {}

    void DPUTransferCompletion::request_completed(void)
    {
      log_dpudma.info() << "dpu memcpy complete: xd=" << std::hex << xd->guid << std::dec
                        << " read=" << read_port_idx << "/" << read_offset
                        << " write=" << write_port_idx << "/" << write_offset
                        << " bytes=" << write_size;
      if(read_port_idx >= 0)
        xd->update_bytes_read(read_port_idx, read_offset, read_size);
      if(write_port_idx >= 0)
        xd->update_bytes_write(write_port_idx, write_offset, write_size);
      xd->remove_reference();
      delete this; // TODO: recycle these!
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // class DPUfillXferDes

    DPUfillXferDes::DPUfillXferDes(uintptr_t _dma_op, Channel *_channel,
                                   NodeID _launch_node, XferDesID _guid,
                                   const std::vector<XferDesPortInfo> &inputs_info,
                                   const std::vector<XferDesPortInfo> &outputs_info,
                                   int _priority, const void *_fill_data,
                                   size_t _fill_size, size_t _fill_total)
      : XferDes(_dma_op, _channel, _launch_node, _guid, inputs_info, outputs_info,
                _priority, _fill_data, _fill_size)
      , pattern(0)
      , pattern_bytes(0)
    {
      kind = XFER_DPU_IN_MRAM;

      // no direct input data for us, but we know how much data to produce
      //  (in case the output is an intermediate buffer)
      assert(input_control.control_port_idx == -1);
      input_control.current_io_port = -1;
      input_control.remaining_count = _fill_total;
      input_control.eos_received = true;
    }

    long DPUfillXferDes::get_requests(Request **requests, long nr)
    {
      // unused
      assert(0);
      return 0;
    }

    DPUfillXferDes::~DPUfillXferDes(void) { free(pattern); }

    bool DPUfillXferDes::progress_xd(DPUfillChannel *channel, TimeLimit work_until)
    {
      bool did_work = false;
      ReadSequenceCache rseqcache(this, MEGABYTE);
      WriteSequenceCache wseqcache(this, MEGABYTE);

      DPU *dpu = channel->dpu;

      // fill_data repeated, reused for every chunk (freed with the xd, after all
      //  completions have released their references)
      if(!pattern) {
        size_t reps = std::max<size_t>(1, PATTERN_BYTES / fill_size);
        pattern_bytes = reps * fill_size;
        pattern = static_cast<char *>(malloc(pattern_bytes));
        for(size_t i = 0; i < reps; i++)
          memcpy(pattern + (i * fill_size), fill_data, fill_size);
      }

      while(true) {
        size_t min_xfer_size = 4096; // TODO: make controllable
        size_t max_bytes = get_addresses(min_xfer_size, &rseqcache);
        if(max_bytes == 0) {
          break;
        }

        XferPort *out_port = 0;
        size_t out_span_start = 0;
        if(output_control.current_io_port >= 0) {
          out_port = &output_ports[output_control.current_io_port];
          out_span_start = out_port->local_bytes_total;
        }

        bool done = false;

        size_t total_bytes = 0;
        if(out_port != 0) {
          log_xd.info() << "dpufill chunk: min=" << min_xfer_size << " max=" << max_bytes;

          while(total_bytes < max_bytes) {
            AddressListCursor &out_alc = out_port->addrcursor;

            uint64_t rank_addr = out_alc.get_offset();
            size_t bytes = std::min(out_alc.remaining(0), max_bytes - total_bytes);
#ifdef DEBUG_REALM
            assert((bytes % fill_size) == 0);
#endif
            for(size_t done_bytes = 0; done_bytes < bytes; done_bytes += pattern_bytes)
              dpu->enqueue_copy(DPU_XFER_TO_DPU, pattern, rank_addr + done_bytes,
                                std::min(pattern_bytes, bytes - done_bytes));

            out_alc.advance(0, bytes);
            total_bytes += bytes;

            // stop if it's been too long, but make sure we do at least the
            //  minimum number of bytes
            if((total_bytes >= min_xfer_size) && work_until.is_expired())
              break;
          }

          // however many fills/copies we submitted, put in a single fence that
          //  will tell us that they're all done
          add_reference(); // released by transfer completion
          dpu->stream->add_notification(
              new DPUTransferCompletion(this, -1, 0, 0, output_control.current_io_port,
                                        out_span_start, total_bytes));
        }

        done = record_address_consumption(total_bytes, total_bytes);

        did_work = true;

        if(done || work_until.is_expired())
          break;
      }

      rseqcache.flush();

      return did_work;
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // class DPUfillChannel

    DPUfillChannel::DPUfillChannel(DPU *_dpu, BackgroundWorkManager *bgwork)
      : SingleXDQChannel<DPUfillChannel, DPUfillXferDes>(bgwork, XFER_DPU_IN_MRAM,
                                                         "fill channel")
      // stringbuilder() << "upmem fill channel (dpu=" << _dpu->info->index << ")")
      , dpu(_dpu)
    {
      std::vector<Memory> local_dpu_mems;
      local_dpu_mems.push_back(dpu->mram->me);

      // look for any other local memories that belong to our context
      const Node &n = get_runtime()->nodes[Network::my_node_id];
      for(std::vector<MemoryImpl *>::const_iterator it = n.memories.begin();
          it != n.memories.end(); ++it) {
        UpmemDeviceMemoryInfo *cdm = (*it)->find_module_specific<UpmemDeviceMemoryInfo>();
        if(!cdm)
          continue;
        if(cdm->device_id != dpu->device_id)
          continue;
        local_dpu_mems.push_back((*it)->me);
      }

      unsigned bw = 300000;          // HACK - estimate at 300 GB/s
      unsigned latency = 250;        // HACK - estimate at 250 ns
      unsigned frag_overhead = 2000; // HACK - estimate at 2 us

      add_path(Memory::NO_MEMORY, local_dpu_mems, bw, latency, frag_overhead,
               XFER_DPU_IN_MRAM)
          .set_max_dim(2);

      xdq.add_to_manager(bgwork);
    }

    XferDes *
    DPUfillChannel::create_xfer_des(uintptr_t dma_op, NodeID launch_node, XferDesID guid,
                                    const std::vector<XferDesPortInfo> &inputs_info,
                                    const std::vector<XferDesPortInfo> &outputs_info,
                                    int priority, XferDesRedopInfo redop_info,
                                    const void *fill_data, size_t fill_size,
                                    size_t fill_total)
    {
      assert(redop_info.id == 0);
      return new DPUfillXferDes(dma_op, this, launch_node, guid, inputs_info,
                                outputs_info, priority, fill_data, fill_size, fill_total);
    }

    long DPUfillChannel::submit(Request **requests, long nr)
    {
      // unused
      assert(0);
      return 0;
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // class DPUreduceXferDes

    DPUreduceXferDes::DPUreduceXferDes(uintptr_t _dma_op, Channel *_channel,
                                       NodeID _launch_node, XferDesID _guid,
                                       const std::vector<XferDesPortInfo> &inputs_info,
                                       const std::vector<XferDesPortInfo> &outputs_info,
                                       int _priority, XferDesRedopInfo _redop_info)
      : XferDes(_dma_op, _channel, _launch_node, _guid, inputs_info, outputs_info,
                _priority, 0, 0)
      , redop_info(_redop_info)
    {
      kind = XFER_DPU_IN_MRAM;
      redop = get_runtime()->reduce_op_table.get(redop_info.id, 0);
      assert(redop);

      DPU *dpu = checked_cast<DPUreduceChannel *>(channel)->dpu;

      // select reduction kernel now - translate to CUfunction if possible
      assert(redop_info.is_exclusive == true &&
             "Reduction on local UPMEM should be exclusive");
      // figure out if its fold or apply
      void *host_proxy =
          (redop_info.is_fold ? redop->upmem_fold_excl_fn : redop->upmem_apply_excl_fn);

      kernel_host_proxy = host_proxy;
      // stream = dpu->get_next_d2d_stream();
      stream = dpu->get_next_stream();
    }

    long DPUreduceXferDes::get_requests(Request **requests, long nr)
    {
      // unused
      assert(0);
      return 0;
    }

    bool DPUreduceXferDes::progress_xd(DPUreduceChannel *channel, TimeLimit work_until)
    {
      bool did_work = false;
      ReadSequenceCache rseqcache(this, MEGABYTE);
      ReadSequenceCache wseqcache(this, MEGABYTE);

      const size_t in_elem_size = redop->sizeof_rhs;
      const size_t out_elem_size =
          (redop_info.is_fold ? redop->sizeof_rhs : redop->sizeof_lhs);
      assert(redop_info.in_place); // TODO: support for out-of-place reduces

      struct KernelArgs {
        uintptr_t dst_base, dst_stride;
        uintptr_t src_base, src_stride;
        uintptr_t count;
      };
      KernelArgs *args = 0; // allocate on demand
      size_t args_size = sizeof(KernelArgs) + redop->sizeof_userdata;

      while(true) {
        size_t min_xfer_size = 4096; // TODO: make controllable
        size_t max_bytes = get_addresses(min_xfer_size, &rseqcache);
        if(max_bytes == 0)
          break;

        XferPort *in_port = 0, *out_port = 0;
        size_t in_span_start = 0, out_span_start = 0;
        if(input_control.current_io_port >= 0) {
          in_port = &input_ports[input_control.current_io_port];
          in_span_start = in_port->local_bytes_total;
        }
        if(output_control.current_io_port >= 0) {
          out_port = &output_ports[output_control.current_io_port];
          out_span_start = out_port->local_bytes_total;
        }

        // have to count in terms of elements, which requires redoing some math
        //  if in/out sizes do not match
        size_t max_elems;
        if(in_elem_size == out_elem_size) {
          max_elems = max_bytes / in_elem_size;
        } else {
          max_elems = std::min(input_control.remaining_count / in_elem_size,
                               output_control.remaining_count / out_elem_size);
          if(in_port != 0) {
            max_elems =
                std::min(max_elems, in_port->addrlist.bytes_pending() / in_elem_size);
            if(in_port->peer_guid != XFERDES_NO_GUID) {
              size_t read_bytes_avail = in_port->seq_remote.span_exists(
                  in_port->local_bytes_total, (max_elems * in_elem_size));
              max_elems = std::min(max_elems, (read_bytes_avail / in_elem_size));
            }
          }
          if(out_port != 0) {
            max_elems =
                std::min(max_elems, out_port->addrlist.bytes_pending() / out_elem_size);
            // no support for reducing into an intermediate buffer
            assert(out_port->peer_guid == XFERDES_NO_GUID);
          }
        }

        size_t total_elems = 0;
        if(in_port != 0) {
          if(out_port != 0) {
            // input and output both exist - transfer what we can
            log_xd.info() << "dpureduce chunk: min=" << min_xfer_size
                          << " max_elems=" << max_elems;

            uintptr_t in_base =
                reinterpret_cast<uintptr_t>(in_port->mem->get_direct_ptr(0, 0));
            uintptr_t out_base =
                reinterpret_cast<uintptr_t>(out_port->mem->get_direct_ptr(0, 0));

            while(total_elems < max_elems) {
              AddressListCursor &in_alc = in_port->addrcursor;
              AddressListCursor &out_alc = out_port->addrcursor;

              uintptr_t in_offset = in_alc.get_offset();
              uintptr_t out_offset = out_alc.get_offset();

              // the reported dim is reduced for partially consumed address
              //  ranges - whatever we get can be assumed to be regular
              int in_dim = in_alc.get_dim();
              int out_dim = out_alc.get_dim();

              // the current reduction op interface can reduce multiple elements
              //  with a fixed address stride, which looks to us like either
              //  1D (stride = elem_size), or 2D with 1 elem/line

              size_t icount = in_alc.remaining(0) / in_elem_size;
              size_t ocount = out_alc.remaining(0) / out_elem_size;
              size_t istride, ostride;
              if((in_dim > 1) && (icount == 1)) {
                in_dim = 2;
                icount = in_alc.remaining(1);
                istride = in_alc.get_stride(1);
              } else {
                in_dim = 1;
                istride = in_elem_size;
              }
              if((out_dim > 1) && (ocount == 1)) {
                out_dim = 2;
                ocount = out_alc.remaining(1);
                ostride = out_alc.get_stride(1);
              } else {
                out_dim = 1;
                ostride = out_elem_size;
              }

              size_t elems_left = max_elems - total_elems;
              size_t elems = std::min(std::min(icount, ocount), elems_left);
              assert(elems > 0);

              // allocate kernel arg structure if this is our first call
              if(!args) {
                args = static_cast<KernelArgs *>(alloca(args_size));
                if(redop->sizeof_userdata)
                  memcpy(args + 1, redop->userdata, redop->sizeof_userdata);
              }

              args->dst_base = out_base + out_offset;
              args->dst_stride = ostride;
              args->src_base = in_base + in_offset;
              args->src_stride = istride;
              args->count = elems;

              // size_t threads_per_block = 256;
              // size_t blocks_per_grid = 1 + ((elems - 1) / threads_per_block);

              {
                void *src_ptr = (void *)args->src_base;
                void *src_device = src_ptr;

                // CHECK_HIP( hipHostGetDevicePointer((void **)&src_device, src_ptr, 0) );

                void *params[] = {&args->dst_base,   &args->dst_stride, &src_device,
                                  &args->src_stride, &args->count,      args + 1};

                // funct(args->dst_base, args->dst_stride, (uintptr_t)src_device,
                //       args->src_stride, args->count, *(int *)(args + 1)

                dpu_set_t dpu_proc;
                DPU_FOREACH(*stream->get_stream(), dpu_proc)
                {
                  CHECK_UPMEM(dpu_prepare_xfer(dpu_proc, params));
                }
                CHECK_UPMEM(dpu_push_xfer(*stream->get_stream(), DPU_XFER_TO_DPU,
                                          "REDUCE_ARGS", 0, args_size, DPU_XFER_DEFAULT));

                DPU_FOREACH(*stream->get_stream(), dpu_proc)
                {
                  CHECK_UPMEM(dpu_prepare_xfer(dpu_proc, (void *)kernel_host_proxy));
                }
                CHECK_UPMEM(dpu_push_xfer(*stream->get_stream(), DPU_XFER_TO_DPU,
                                          "REDUCE_FUNCTION", 0, sizeof(kernel_host_proxy),
                                          DPU_XFER_DEFAULT));

                CHECK_UPMEM(dpu_launch(*stream->get_stream(), DPU_ASYNCHRONOUS));

                // CHECK_HIP( hipGetDevice(&orig_device) );
                // CHECK_HIP( hipSetDevice(channel->dpu->info->index) );
                // CHECK_HIP( hipLaunchKernel(kernel_host_proxy,
                //                          dim3(blocks_per_grid),
                //                          dim3(threads_per_block),
                //                          params,
                //                          0 /*sharedmem*/,
                //                          stream->get_stream()) );
                // CHECK_HIP( hipSetDevice(orig_device) );

                // insert fence to track completion of reduction kernel
                add_reference(); // released by transfer completion
                stream->add_notification(new DPUTransferCompletion(
                    this, input_control.current_io_port, in_span_start,
                    elems * in_elem_size, output_control.current_io_port, out_span_start,
                    elems * out_elem_size));
              }

              in_span_start += elems * in_elem_size;
              out_span_start += elems * out_elem_size;

              in_alc.advance(in_dim - 1, elems * ((in_dim == 1) ? in_elem_size : 1));
              out_alc.advance(out_dim - 1, elems * ((out_dim == 1) ? out_elem_size : 1));

#ifdef DEBUG_REALM
              assert(elems <= elems_left);
#endif
              total_elems += elems;

              // stop if it's been too long, but make sure we do at least the
              //  minimum number of bytes
              if(((total_elems * in_elem_size) >= min_xfer_size) &&
                 work_until.is_expired())
                break;
            }
          } else {
            // input but no output, so skip input bytes
            total_elems = max_elems;
            in_port->addrcursor.skip_bytes(total_elems * in_elem_size);

            rseqcache.add_span(input_control.current_io_port, in_span_start,
                               total_elems * in_elem_size);
            in_span_start += total_elems * in_elem_size;
          }
        } else {
          if(out_port != 0) {
            // output but no input, so skip output bytes
            total_elems = max_elems;
            out_port->addrcursor.skip_bytes(total_elems * out_elem_size);

            wseqcache.add_span(output_control.current_io_port, out_span_start,
                               total_elems * out_elem_size);
            out_span_start += total_elems * out_elem_size;
          } else {
            // skipping both input and output is possible for simultaneous
            //  gather+scatter
            total_elems = max_elems;
          }
        }

        bool done = record_address_consumption(total_elems * in_elem_size,
                                               total_elems * out_elem_size);

        did_work = true;

        if(done || work_until.is_expired())
          break;
      }

      rseqcache.flush();
      wseqcache.flush();

      return did_work;
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // class DPUreduceChannel

    DPUreduceChannel::DPUreduceChannel(DPU *_dpu, BackgroundWorkManager *bgwork)
      : SingleXDQChannel<DPUreduceChannel, DPUreduceXferDes>(bgwork, XFER_DPU_IN_MRAM,
                                                             " ") // _dpu->info->index
      , dpu(_dpu)
    {
      std::vector<Memory> local_dpu_mems;
      local_dpu_mems.push_back(dpu->mram->me);

      std::vector<Memory> peer_dpu_mems;
      peer_dpu_mems.insert(peer_dpu_mems.end(), dpu->peer_mram.begin(),
                           dpu->peer_mram.end());

      // look for any other local memories that belong to our context or
      //  peer-able contexts
      const Node &n = get_runtime()->nodes[Network::my_node_id];
      for(std::vector<MemoryImpl *>::const_iterator it = n.memories.begin();
          it != n.memories.end(); ++it) {
        UpmemDeviceMemoryInfo *cdm = (*it)->find_module_specific<UpmemDeviceMemoryInfo>();
        if(!cdm)
          continue;
        if(cdm->device_id == dpu->device_id) {
          local_dpu_mems.push_back((*it)->me);
        } else {
          // if the other context is associated with a dpu and we've got peer
          //  access, use it
          // TODO: add option to enable peer access at this point?  might be
          //  expensive...
          if(cdm->dpu && (dpu->info->peers.count(cdm->dpu->info->device) > 0))
            peer_dpu_mems.push_back((*it)->me);
        }
      }

      std::vector<Memory> mapped_cpu_mems;
      mapped_cpu_mems.insert(mapped_cpu_mems.end(), dpu->pinned_sysmems.begin(),
                             dpu->pinned_sysmems.end());

      // intra-FB reduction
      {
        unsigned bw = 100000;          // HACK - estimate at 100 GB/s
        unsigned latency = 250;        // HACK - estimate at 250 ns
        unsigned frag_overhead = 2000; // HACK - estimate at 2 us

        add_path(local_dpu_mems, local_dpu_mems, bw, latency, frag_overhead,
                 XFER_DPU_IN_MRAM)
            .allow_redops();
      }

      // zero-copy to FB (no need for intermediate buffer in FB)
      {
        unsigned bw = 10000;           // HACK - estimate at 10 GB/s
        unsigned latency = 1000;       // HACK - estimate at 1 us
        unsigned frag_overhead = 2000; // HACK - estimate at 2 us

        add_path(mapped_cpu_mems, local_dpu_mems, bw, latency, frag_overhead,
                 XFER_DPU_TO_MRAM)
            .allow_redops();
      }

      // unlike normal cuda p2p copies where we want to push from the source,
      //  reductions are always sent to the destination memory's dpu to keep the
      //  RMW loop as tight as possible
      {
        unsigned bw = 50000;           // HACK - estimate at 50 GB/s
        unsigned latency = 2000;       // HACK - estimate at 1 us
        unsigned frag_overhead = 2000; // HACK - estimate at 2 us

        add_path(peer_dpu_mems, local_dpu_mems, bw, latency, frag_overhead,
                 XFER_DPU_PEER_MRAM)
            .allow_redops();
      }

      xdq.add_to_manager(bgwork);
    }

    /*static*/ bool DPUreduceChannel::is_dpu_redop(ReductionOpID redop_id)
    {
      if(redop_id == 0)
        return false;

      ReductionOpUntyped *redop = get_runtime()->reduce_op_table.get(redop_id, 0);
      assert(redop);

      // there's four different kernels, but they should be all or nothing, so
      //  just check one
      if(!redop->upmem_apply_excl_fn)
        return false;

      return true;
    }

    uint64_t DPUreduceChannel::supports_path(
        ChannelCopyInfo channel_copy_info, CustomSerdezID src_serdez_id,
        CustomSerdezID dst_serdez_id, ReductionOpID redop_id, size_t total_bytes,
        const std::vector<size_t> *src_frags, const std::vector<size_t> *dst_frags,
        XferDesKind *kind_ret /*= 0*/, unsigned *bw_ret /*= 0*/,
        unsigned *lat_ret /*= 0*/)
    {
      // first check that we have a reduction op (if not, we want the cudamemcpy
      //   path to pick this up instead) and that it has cuda kernels available
      if(!is_dpu_redop(redop_id))
        return 0;

      // then delegate to the normal supports_path logic
      return Channel::supports_path(channel_copy_info, src_serdez_id, dst_serdez_id,
                                    redop_id, total_bytes, src_frags, dst_frags, kind_ret,
                                    bw_ret, lat_ret);
    }

    RemoteChannelInfo *DPUreduceChannel::construct_remote_info() const
    {
      return new DPUreduceRemoteChannelInfo(node, kind, reinterpret_cast<uintptr_t>(this),
                                            paths);
    }

    XferDes *DPUreduceChannel::create_xfer_des(
        uintptr_t dma_op, NodeID launch_node, XferDesID guid,
        const std::vector<XferDesPortInfo> &inputs_info,
        const std::vector<XferDesPortInfo> &outputs_info, int priority,
        XferDesRedopInfo redop_info, const void *fill_data, size_t fill_size,
        size_t fill_total)
    {
      assert(fill_size == 0);
      return new DPUreduceXferDes(dma_op, this, launch_node, guid, inputs_info,
                                  outputs_info, priority, redop_info);
    }

    long DPUreduceChannel::submit(Request **requests, long nr)
    {
      // unused
      assert(0);
      return 0;
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // class DPUreduceRemoteChannelInfo
    //

    DPUreduceRemoteChannelInfo::DPUreduceRemoteChannelInfo(
        NodeID _owner, XferDesKind _kind, uintptr_t _remote_ptr,
        const std::vector<Channel::SupportedPath> &_paths)
      : SimpleRemoteChannelInfo(_owner, _kind, _remote_ptr, _paths)
    {}

    RemoteChannel *DPUreduceRemoteChannelInfo::create_remote_channel()
    {
      DPUreduceRemoteChannel *rc = new DPUreduceRemoteChannel(remote_ptr);
      rc->node = owner;
      rc->kind = kind;
      rc->paths.swap(paths);
      return rc;
    }

    // these templates can go here because they're only used by the helper below
    template <typename S>
    bool DPUreduceRemoteChannelInfo::serialize(S &serializer) const
    {
      return ((serializer << owner) && (serializer << kind) &&
              (serializer << remote_ptr) && (serializer << paths));
    }

    template <typename S>
    /*static*/ RemoteChannelInfo *
    DPUreduceRemoteChannelInfo::deserialize_new(S &deserializer)
    {
      NodeID owner;
      XferDesKind kind;
      uintptr_t remote_ptr;
      std::vector<Channel::SupportedPath> paths;

      if((deserializer >> owner) && (deserializer >> kind) &&
         (deserializer >> remote_ptr) && (deserializer >> paths)) {
        return new DPUreduceRemoteChannelInfo(owner, kind, remote_ptr, paths);
      } else {
        return 0;
      }
    }

    /*static*/ Serialization::PolymorphicSerdezSubclass<RemoteChannelInfo,
                                                        DPUreduceRemoteChannelInfo>
        DPUreduceRemoteChannelInfo::serdez_subclass;

    ////////////////////////////////////////////////////////////////////////
    //
    // class DPUreduceRemoteChannel
    //

    DPUreduceRemoteChannel::DPUreduceRemoteChannel(uintptr_t _remote_ptr)
      : RemoteChannel(_remote_ptr)
    {}

    uint64_t DPUreduceRemoteChannel::supports_path(
        ChannelCopyInfo channel_copy_info, CustomSerdezID src_serdez_id,
        CustomSerdezID dst_serdez_id, ReductionOpID redop_id, size_t total_bytes,
        const std::vector<size_t> *src_frags, const std::vector<size_t> *dst_frags,
        XferDesKind *kind_ret /*= 0*/, unsigned *bw_ret /*= 0*/,
        unsigned *lat_ret /*= 0*/)
    {
      // check first that we have a reduction op (if not, we want the cudamemcpy
      //   path to pick this up instead) and that it has cuda kernels available
      if(!DPUreduceChannel::is_dpu_redop(redop_id))
        return 0;

      // then delegate to the normal supports_path logic
      return Channel::supports_path(channel_copy_info, src_serdez_id, dst_serdez_id,
                                    redop_id, total_bytes, src_frags, dst_frags, kind_ret,
                                    bw_ret, lat_ret);
    }

  }; // namespace Upmem
};   // namespace Realm