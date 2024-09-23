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

#ifndef REALM_UPMEM_DMA_H
#define REALM_UPMEM_DMA_H

#include "realm/upmem/upmem_internal.h"

#include "realm/transfer/channel.h"
#include "realm/transfer/ib_memory.h"
#include "realm/transfer/lowlevel_dma.h"

#include "realm/bgwork.h"
#include "realm/network.h"

typedef void (*proxy_function)(uintptr_t side1_base, uintptr_t side1_stride,
                               uintptr_t side2_base, uintptr_t side2_stride, size_t count,
                               int redop);

namespace Realm {
  namespace Upmem {
    // forward declaration
    // internal.h
    class DPU;

    class DPUTransferCompletion : public DPUCompletionNotification {
    public:
      DPUTransferCompletion(XferDes *_xd, int _read_port_idx, size_t _read_offset,
                            size_t _read_size, int _write_port_idx, size_t _write_offset,
                            size_t _write_size);

      virtual void request_completed(void);

    protected:
      XferDes *xd;
      int read_port_idx;
      size_t read_offset, read_size;
      int write_port_idx;
      size_t write_offset, write_size;
    };

    class DPUChannel;

    class DPUXferDes : public XferDes {
    public:
      DPUXferDes(uintptr_t _dma_op, Channel *_channel, NodeID _launch_node,
                 XferDesID _guid, const std::vector<XferDesPortInfo> &inputs_info,
                 const std::vector<XferDesPortInfo> &outputs_info, int _priority);

      long get_requests(Request **requests, long nr);

      bool progress_xd(DPUChannel *channel, TimeLimit work_until);

    private:
      std::vector<DPU *> src_dpus, dst_dpus;
      std::vector<bool> dst_is_ipc;
    };

    class DPUChannel : public SingleXDQChannel<DPUChannel, DPUXferDes> {
    public:
      DPUChannel(DPU *_src_dpu, XferDesKind _kind, BackgroundWorkManager *bgwork);
      ~DPUChannel();

      // multi-threading of cuda copies for a given device is disabled by
      //  default (can be re-enabled with -cuda:mtdma 1)
      static const bool is_ordered = true;

      virtual XferDes *create_xfer_des(uintptr_t dma_op, NodeID launch_node,
                                       XferDesID guid,
                                       const std::vector<XferDesPortInfo> &inputs_info,
                                       const std::vector<XferDesPortInfo> &outputs_info,
                                       int priority, XferDesRedopInfo redop_info,
                                       const void *fill_data, size_t fill_size,
                                       size_t fill_total);

      long submit(Request **requests, long nr);

    private:
      DPU *src_dpu;
      // std::deque<Request*> pending_copies;
    };

    class DPUfillChannel;

    class DPUfillXferDes : public XferDes {
    public:
      DPUfillXferDes(uintptr_t _dma_op, Channel *_channel, NodeID _launch_node,
                     XferDesID _guid, const std::vector<XferDesPortInfo> &inputs_info,
                     const std::vector<XferDesPortInfo> &outputs_info, int _priority,
                     const void *_fill_data, size_t _fill_size, size_t _fill_total);

      long get_requests(Request **requests, long nr);

      bool progress_xd(DPUfillChannel *channel, TimeLimit work_until);

    protected:
      size_t reduced_fill_size;
    };

    class DPUfillChannel : public SingleXDQChannel<DPUfillChannel, DPUfillXferDes> {
    public:
      DPUfillChannel(DPU *_dpu, BackgroundWorkManager *bgwork);

      // multiple concurrent cuda fills ok
      static const bool is_ordered = false;

      virtual XferDes *create_xfer_des(uintptr_t dma_op, NodeID launch_node,
                                       XferDesID guid,
                                       const std::vector<XferDesPortInfo> &inputs_info,
                                       const std::vector<XferDesPortInfo> &outputs_info,
                                       int priority, XferDesRedopInfo redop_info,
                                       const void *fill_data, size_t fill_size,
                                       size_t fill_total);

      long submit(Request **requests, long nr);

    protected:
      friend class DPUfillXferDes;

      DPU *dpu;
    };

    class DPUreduceChannel;

    class DPUreduceXferDes : public XferDes {
    public:
      DPUreduceXferDes(uintptr_t _dma_op, Channel *_channel, NodeID _launch_node,
                       XferDesID _guid, const std::vector<XferDesPortInfo> &inputs_info,
                       const std::vector<XferDesPortInfo> &outputs_info, int _priority,
                       XferDesRedopInfo _redop_info);

      long get_requests(Request **requests, long nr);

      bool progress_xd(DPUreduceChannel *channel, TimeLimit work_until);

    protected:
      XferDesRedopInfo redop_info;
      const ReductionOpUntyped *redop;
      const void *kernel_host_proxy;
      DPUStream *stream;
    };

    class DPUreduceChannel : public SingleXDQChannel<DPUreduceChannel, DPUreduceXferDes> {
    public:
      DPUreduceChannel(DPU *_DPU, BackgroundWorkManager *bgwork);

      // multiple concurrent cuda reduces ok
      static const bool is_ordered = false;

      // helper method here so that DPUreduceRemoteChannel can use it too
      static bool is_dpu_redop(ReductionOpID redop_id);

      // override this because we have to be picky about which reduction ops
      //  we support
      virtual uint64_t
      supports_path(ChannelCopyInfo channel_copy_info, CustomSerdezID src_serdez_id,
                    CustomSerdezID dst_serdez_id, ReductionOpID redop_id,
                    size_t total_bytes, const std::vector<size_t> *src_frags,
                    const std::vector<size_t> *dst_frags, XferDesKind *kind_ret = 0,
                    unsigned *bw_ret = 0, unsigned *lat_ret = 0);

      virtual RemoteChannelInfo *construct_remote_info() const;

      virtual XferDes *create_xfer_des(uintptr_t dma_op, NodeID launch_node,
                                       XferDesID guid,
                                       const std::vector<XferDesPortInfo> &inputs_info,
                                       const std::vector<XferDesPortInfo> &outputs_info,
                                       int priority, XferDesRedopInfo redop_info,
                                       const void *fill_data, size_t fill_size,
                                       size_t fill_total);

      long submit(Request **requests, long nr);

    protected:
      friend class DPUreduceXferDes;

      DPU *dpu;
    };

    class DPUreduceRemoteChannelInfo : public SimpleRemoteChannelInfo {
    public:
      DPUreduceRemoteChannelInfo(NodeID _owner, XferDesKind _kind, uintptr_t _remote_ptr,
                                 const std::vector<Channel::SupportedPath> &_paths);

      virtual RemoteChannel *create_remote_channel();

      template <typename S>
      bool serialize(S &serializer) const;

      template <typename S>
      static RemoteChannelInfo *deserialize_new(S &deserializer);

    protected:
      static Serialization::PolymorphicSerdezSubclass<RemoteChannelInfo,
                                                      DPUreduceRemoteChannelInfo>
          serdez_subclass;
    };

    class DPUreduceRemoteChannel : public RemoteChannel {
      friend class DPUreduceRemoteChannelInfo;

      DPUreduceRemoteChannel(uintptr_t _remote_ptr);

      virtual uint64_t
      supports_path(ChannelCopyInfo channel_copy_info, CustomSerdezID src_serdez_id,
                    CustomSerdezID dst_serdez_id, ReductionOpID redop_id,
                    size_t total_bytes, const std::vector<size_t> *src_frags,
                    const std::vector<size_t> *dst_frags, XferDesKind *kind_ret = 0,
                    unsigned *bw_ret = 0, unsigned *lat_ret = 0);
    };

  }; // namespace Upmem
};   // namespace Realm

#endif
