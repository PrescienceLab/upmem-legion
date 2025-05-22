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

#define XFER_SYNC_TYPE DPU_XFER_ASYNC

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
      std::string memcpy_kind;

      ReadSequenceCache rseqcache(this,  MEGABYTE);
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
        bool out_is_ipc = false;
        int out_ipc_index = -1;
        if(input_control.current_io_port >= 0) {
          in_port = &input_ports[input_control.current_io_port];
          in_span_start = in_port->local_bytes_total;
          in_dpu = src_dpus[input_control.current_io_port];
        }
        if(output_control.current_io_port >= 0) {
          out_port = &output_ports[output_control.current_io_port];
          out_span_start = out_port->local_bytes_total;
          out_dpu = dst_dpus[output_control.current_io_port];
          out_is_ipc = dst_is_ipc[output_control.current_io_port];
        }

        size_t total_bytes = 0;
        if(in_port != 0) {
          if(out_port != 0) {
            // input and output both exist - transfer what we can
            log_xd.info() << "upmem memcpy chunk: min=" << min_xfer_size
                          << " max=" << max_bytes;

            uintptr_t in_base =
                reinterpret_cast<uintptr_t>(in_port->mem->get_direct_ptr(0, 0));
            uintptr_t out_base;
            const DPU::UpmemIpcMapping *out_mapping = 0;
            if(out_is_ipc) {
              out_mapping = in_dpu->find_ipc_mapping(out_port->mem->me);
              assert(out_mapping);
              out_base = out_mapping->local_base;
            } else
              out_base = reinterpret_cast<uintptr_t>(out_port->mem->get_direct_ptr(0, 0));

            DPUStream *stream = NULL;

            if(in_dpu) {
              if(out_dpu == in_dpu) {
                memcpy_kind = "d2d";
              } else if(out_mapping) {
                memcpy_kind = "ipc";
              } else if(!out_dpu) {
                memcpy_kind = "d2h";
              } else {
                memcpy_kind = "p2p";
              }
            } else {
              assert(out_dpu);
              memcpy_kind = "h2d";
            }

            size_t bytes_to_fence = 0;

            while(total_bytes < max_bytes) {
              AddressListCursor &in_alc = in_port->addrcursor;
              AddressListCursor &out_alc = out_port->addrcursor;

              uintptr_t in_offset = in_alc.get_offset();
              uintptr_t out_offset = out_alc.get_offset();

              // the reported dim is reduced for partially consumed address
              //  ranges - whatever we get can be assumed to be regular
              int in_dim = in_alc.get_dim();
              int out_dim = out_alc.get_dim();

              size_t bytes = 0;
              size_t bytes_left = max_bytes - total_bytes;

              // limit transfer size for host<->device copies
              if((bytes_left > (32 * MEGABYTE)) &&
                 (!in_dpu || (!out_dpu && (out_ipc_index == -1))))
                bytes_left = 32 * MEGABYTE;

              assert(in_dim > 0);
              assert(out_dim > 0);

              size_t icount = in_alc.remaining(0);
              size_t ocount = out_alc.remaining(0);

              // contig bytes is always the min of the first dimensions
              size_t contig_bytes = std::min(std::min(icount, ocount), bytes_left);

              // catch simple 1D case first
              if((contig_bytes == bytes_left) ||
                 ((contig_bytes == icount) && (in_dim == 1)) ||
                 ((contig_bytes == ocount) && (out_dim == 1))) {
                bytes = contig_bytes;

                // check rate limit on stream
                if(!stream->ok_to_submit_copy(bytes, this)) {
                  break;
                }

                // grr...  prototypes of these differ slightly...
                DPUMemcpyKind copy_type;
                size_t baseoffset_host = 0;
                size_t baseoffset_dpu = 0;

                uint64_t idx_dpu = 0;
                if(in_dpu) {
                  if(out_dpu == in_dpu || (out_ipc_index >= 0)) {
                    printf("device to device not currently supported\n");
                  } else if(!out_dpu) {
                    copy_type = DPU_XFER_FROM_DPU;
                    stream = in_dpu->stream;
                    baseoffset_dpu = DPU_REALM_ADDRESS((in_base + in_offset)); // input = dpu
                    baseoffset_host = out_base + out_offset; // output = host
                  }
                } else {
                  copy_type = DPU_XFER_TO_DPU;
                  stream = out_dpu->stream;
                  baseoffset_host = in_base + in_offset; // input = host
                  baseoffset_dpu = DPU_REALM_ADDRESS((out_base + out_offset)); // output = dpu
                }

                CHECK_UPMEM(
                    dpu_prepare_xfer(*(stream->get_stream()), (void *)(baseoffset_host)));
                CHECK_UPMEM(dpu_push_xfer(*(stream->get_stream()), copy_type,
                                          DPU_MRAM_HEAP_POINTER_NAME, baseoffset_dpu,
                                          bytes, XFER_SYNC_TYPE));

                // CHECK_HIP(
                //     hipMemcpyAsync(reinterpret_cast<void *>(out_base + out_offset),
                //                      reinterpret_cast<const void *>(in_base +
                //                      in_offset), bytes, copy_type,
                //                      (stream->get_stream())));

                log_dpudma.info()
                    << "dpu memcpy: dst=" << std::hex << (out_base + out_offset)
                    << " src=" << (in_base + in_offset) << std::dec << " bytes=" << bytes
                    << " stream=" << stream << " kind=" << memcpy_kind;

                in_alc.advance(0, bytes);
                out_alc.advance(0, bytes);

                bytes_to_fence += bytes;
                // TODO: fence on a threshold
              } else {
                // grow to a 2D copy
                int id;
                int iscale;
                uintptr_t in_lstride;
                if(contig_bytes < icount) {
                  // second input dim comes from splitting first
                  id = 0;
                  in_lstride = contig_bytes;
                  size_t ilines = icount / contig_bytes;
                  if((ilines * contig_bytes) != icount)
                    in_dim = 1; // leftover means we can't go beyond this
                  icount = ilines;
                  iscale = contig_bytes;
                } else {
                  assert(in_dim > 1);
                  id = 1;
                  icount = in_alc.remaining(id);
                  in_lstride = in_alc.get_stride(id);
                  iscale = 1;
                }

                int od;
                int oscale;
                uintptr_t out_lstride;
                if(contig_bytes < ocount) {
                  // second output dim comes from splitting first
                  od = 0;
                  out_lstride = contig_bytes;
                  size_t olines = ocount / contig_bytes;
                  if((olines * contig_bytes) != ocount)
                    out_dim = 1; // leftover means we can't go beyond this
                  ocount = olines;
                  oscale = contig_bytes;
                } else {
                  assert(out_dim > 1);
                  od = 1;
                  ocount = out_alc.remaining(od);
                  out_lstride = out_alc.get_stride(od);
                  oscale = 1;
                }

                size_t lines =
                    std::min(std::min(icount, ocount), bytes_left / contig_bytes);

                // see if we need to stop at 2D
                if(((contig_bytes * lines) == bytes_left) ||
                   ((lines == icount) && (id == (in_dim - 1))) ||
                   ((lines == ocount) && (od == (out_dim - 1)))) {
                  bytes = contig_bytes * lines;

                  // check rate limit on stream
                  if(!stream->ok_to_submit_copy(bytes, this))
                    break;

                  DPUMemcpyKind copy_type;
                  size_t baseoffset_dpu, baseoffset_host;
                  uint64_t idx_dpu = 0;
                  if(in_dpu) {
                    if(out_dpu == in_dpu || (out_ipc_index >= 0)) {
                      printf("device to device not currently supported\n");
                    } else if(!out_dpu) {
                      copy_type = DPU_XFER_FROM_DPU;
                      stream = in_dpu->stream;
                      baseoffset_dpu = DPU_REALM_ADDRESS((out_base + out_offset)); // input = dpu
                      baseoffset_host = in_base + in_offset; // output = host
                    }
                  } else {
                    copy_type = DPU_XFER_TO_DPU;
                    stream = out_dpu->stream; 
                    baseoffset_host = in_base + in_offset; // input = host
                    baseoffset_dpu = DPU_REALM_ADDRESS((out_base + out_offset)); // output = dpu
                  }

                  const void *src = reinterpret_cast<const void *>(baseoffset_host);


                  log_dpudma.info()
                      << "dpu memcpy 2d: dst=" << std::hex << (baseoffset_dpu) << std::dec
                      << "+" << out_lstride << " src=" << std::hex << (baseoffset_host)
                      << std::dec << "+" << in_lstride << " bytes=" << bytes
                      << " lines=" << lines << " stream=" << stream
                      << " kind=" << memcpy_kind;

                  CHECK_UPMEM(dpu_prepare_xfer(*(stream->get_stream()), (void *)src));
                  CHECK_UPMEM(dpu_push_xfer(*(stream->get_stream()), copy_type,
                                            DPU_MRAM_HEAP_POINTER_NAME, baseoffset_dpu,
                                            lines * contig_bytes, XFER_SYNC_TYPE));

                  // CHECK_HIP(hipMemcpy2DAsync(dst, out_lstride, src, in_lstride,
                  //                                contig_bytes, lines, copy_type,
                  //                                (stream->get_stream())));

                  in_alc.advance(id, lines * iscale);
                  out_alc.advance(od, lines * oscale);

                  bytes_to_fence += bytes;
                  // TODO: fence on a threshold
                } else {
                  uintptr_t in_pstride;
                  if(lines < icount) {
                    // third input dim comes from splitting current
                    in_pstride = in_lstride * lines;
                    size_t iplanes = icount / lines;
                    // check for leftovers here if we go beyond 3D!
                    icount = iplanes;
                    iscale *= lines;
                  } else {
                    id++;
                    assert(in_dim > id);
                    icount = in_alc.remaining(id);
                    in_pstride = in_alc.get_stride(id);
                    iscale = 1;
                  }

                  uintptr_t out_pstride;
                  if(lines < ocount) {
                    // third output dim comes from splitting current
                    out_pstride = out_lstride * lines;
                    size_t oplanes = ocount / lines;
                    // check for leftovers here if we go beyond 3D!
                    ocount = oplanes;
                    oscale *= lines;
                  } else {
                    od++;
                    assert(out_dim > od);
                    ocount = out_alc.remaining(od);
                    out_pstride = out_alc.get_stride(od);
                    oscale = 1;
                  }

                  size_t planes = std::min(std::min(icount, ocount),
                                           (bytes_left / (contig_bytes * lines)));

                  // a cuMemcpy3DAsync appears to be unrolled on the host in the
                  //  driver, so we'll do the unrolling into 2D copies ourselves,
                  //  allowing us to stop early if we hit the rate limit or a
                  //  timeout
                  DPUMemcpyKind copy_type;
                  size_t baseoffset_dpu, baseoffset_host;

                  size_t act_planes = 0;

                  uint64_t idx_dpu = 0;
                  while(act_planes < planes) {
                    
                    if(in_dpu) {
                      if(out_dpu == in_dpu || (out_ipc_index >= 0)) {
                        printf("device to device not currently supported\n");
                      } else if(!out_dpu) {
                        copy_type = DPU_XFER_FROM_DPU;
                        stream = in_dpu->stream;
                        baseoffset_dpu =
                            DPU_REALM_ADDRESS((out_base + out_offset + (act_planes * out_pstride))); // input = dpu
                        baseoffset_host = in_base + in_offset + (act_planes * in_pstride); // output = host
                      }
                    } else {
                      copy_type = DPU_XFER_TO_DPU;
                      stream = out_dpu->stream;
                      baseoffset_host = in_base + in_offset + (act_planes * in_pstride); // input = host
                      baseoffset_dpu =
                          DPU_REALM_ADDRESS((out_base + out_offset + (act_planes * out_pstride))); // output = dpu
                    }

                    // check rate limit on stream
                    if(!stream->ok_to_submit_copy(contig_bytes * lines, this))
                      break;

                    CHECK_UPMEM(dpu_prepare_xfer(*(stream->get_stream()),
                                                 (void *)baseoffset_host));
                    CHECK_UPMEM(dpu_push_xfer(*(stream->get_stream()), copy_type,
                                              DPU_MRAM_HEAP_POINTER_NAME, baseoffset_dpu,
                                              lines * contig_bytes, XFER_SYNC_TYPE));

                    // CHECK_HIP(hipMemcpy2DAsync(dst, out_lstride, src, in_lstride,
                    //                                contig_bytes, lines, copy_type,
                    //                                (stream->get_stream())));
                    act_planes++;

                    if(work_until.is_expired())
                      break;
                  }

                  if(act_planes == 0)
                    break;

                  log_dpudma.info()
                      << "dpu memcpy 3d: dst=" << std::hex << (out_base + out_offset)
                      << std::dec << "+" << out_lstride << "+" << out_pstride
                      << " src=" << std::hex << (in_base + in_offset) << std::dec << "+"
                      << in_lstride << "+" << in_pstride << " bytes=" << bytes
                      << " lines=" << lines << " planes=" << act_planes
                      << " stream=" << stream << " kind=" << memcpy_kind;

                  bytes = contig_bytes * lines * act_planes;
                  in_alc.advance(id, act_planes * iscale);
                  out_alc.advance(od, act_planes * oscale);

                  bytes_to_fence += bytes;
                  // TODO: fence on a threshold
                }
              }

#ifdef DEBUG_REALM
              assert(bytes <= bytes_left);
#endif
              total_bytes += bytes;

              // stop if it's been too long, but make sure we do at least the
              //  minimum number of bytes
              if((total_bytes >= min_xfer_size) && work_until.is_expired())
                break;
            }

            if(bytes_to_fence > 0) {
              add_reference(); // released by transfer completion
              log_dpudma.info()
                  << "dpu memcpy fence: stream=" << stream << " xd=" << std::hex << guid
                  << std::dec << " bytes=" << total_bytes;
              stream->add_notification(new DPUTransferCompletion(
                  this, input_control.current_io_port, in_span_start, total_bytes,
                  output_control.current_io_port, out_span_start, total_bytes));
              in_span_start += total_bytes;
              out_span_start += total_bytes;
            }
          } else { // out_port == 0
            // input but no output, so skip input bytes
            total_bytes = max_bytes;
            in_port->addrcursor.skip_bytes(total_bytes);

            rseqcache.add_span(input_control.current_io_port, in_span_start, total_bytes);
            in_span_start += total_bytes;
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
            out_span_start += total_bytes;
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

    bool DPUfillXferDes::progress_xd(DPUfillChannel *channel, TimeLimit work_until)
    {
      bool did_work = false;
      ReadSequenceCache rseqcache(this, MEGABYTE);
      WriteSequenceCache wseqcache(this, MEGABYTE);

      DPUStream *stream = channel->dpu->get_next_stream(false);

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
          // input and output both exist - transfer what we can
          log_xd.info() << "dpufill chunk: min=" << min_xfer_size << " max=" << max_bytes;

          uintptr_t out_base =
              reinterpret_cast<uintptr_t>(out_port->mem->get_direct_ptr(0, 0));

          while(total_bytes < max_bytes) {
            AddressListCursor &out_alc = out_port->addrcursor;

            uintptr_t out_offset = out_alc.get_offset();

            // the reported dim is reduced for partially consumed address
            //  ranges - whatever we get can be assumed to be regular
            int out_dim = out_alc.get_dim();

            // more general approach - use strided 2d copies to fill the first
            //  line, and then we can use logarithmic doublings to deal with
            //  multiple lines and/or planes
            size_t bytes = out_alc.remaining(0);
            size_t elems = bytes / fill_size;

#ifdef DEBUG_REALM
            assert((bytes % fill_size) == 0);
#endif

            void *buffer = (void *)malloc(elems * fill_size);

            const char *fill_buffer = reinterpret_cast<const char *>(fill_data);

            for(unsigned int i = 0; i < elems; i++) {
              memcpy((void *)((char *)buffer + fill_size * i), fill_buffer, fill_size);
            }
            
            {
              CHECK_UPMEM(dpu_prepare_xfer(*(stream->get_stream()), buffer));
              CHECK_UPMEM(dpu_push_xfer(*(stream->get_stream()), DPU_XFER_TO_DPU,
                                        DPU_MRAM_HEAP_POINTER_NAME, DPU_REALM_ADDRESS(out_base + out_offset),
                                        elems * fill_size, XFER_SYNC_TYPE));
            }

            // need to make sure the async transfer is done before we free the buffer
            CHECK_UPMEM(dpu_sync(*(stream->get_stream())));
            free(buffer);

            if(out_dim == 1) {
              // all done
              out_alc.advance(0, bytes);
              total_bytes += bytes;
            } else {
              assert(0 && "Not implemented correctly");
              size_t lines = out_alc.remaining(1);
              size_t lstride = out_alc.get_stride(1);

              void *srcDevice = (void *)(out_base + out_offset);

              size_t lines_done = 1; // first line already valid
              while(lines_done < lines) {
                size_t todo = std::min(lines_done, lines - lines_done);
                size_t dstDevice = (out_base + out_offset + (lines_done * lstride));

   
                CHECK_UPMEM(dpu_prepare_xfer(*(stream->get_stream()), (void *)srcDevice));
                CHECK_UPMEM(dpu_push_xfer(*(stream->get_stream()), DPU_XFER_TO_DPU,
                                          DPU_MRAM_HEAP_POINTER_NAME, DPU_REALM_ADDRESS(dstDevice),
                                          bytes * todo, XFER_SYNC_TYPE));

                // CHECK_HIP(hipMemcpy2DAsync(dstDevice, lstride, srcDevice, lstride,
                //                                bytes, todo, , (stream->get_stream())));
                lines_done += todo;
              }

              if(out_dim == 2) {
                out_alc.advance(1, lines);
                total_bytes += bytes * lines;
              } else {
                assert(0 && "Not implemented correctly");
                size_t planes = out_alc.remaining(2);
                size_t pstride = out_alc.get_stride(2);

                for(size_t p = 1; p < planes; p++) {
                  size_t dstDevice = (out_base + out_offset + (p * pstride));

                  CHECK_UPMEM(
                      dpu_prepare_xfer(*(stream->get_stream()), (void *)srcDevice));
                  CHECK_UPMEM(dpu_push_xfer(*(stream->get_stream()), DPU_XFER_TO_DPU,
                                            DPU_MRAM_HEAP_POINTER_NAME, DPU_REALM_ADDRESS(dstDevice),
                                            bytes * lines, XFER_SYNC_TYPE));

                  // CHECK_HIP(hipMemcpy2DAsync(dstDevice, lstride, srcDevice, lstride,
                  //                                bytes, lines, ,
                  //                                (stream->get_stream())));
                }
              }
              break;
            }

            // stop if it's been too long, but make sure we do at least the
            //  minimum number of bytes
            if((total_bytes >= min_xfer_size) && work_until.is_expired())
              break;
          }

          // however many fills/copies we submitted, put in a single fence that
          //  will tell us that they're all done
          add_reference(); // released by transfer completion
          stream->add_notification(
              new DPUTransferCompletion(this, -1, 0, 0, output_control.current_io_port,
                                        out_span_start, total_bytes));
          out_span_start += total_bytes;
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