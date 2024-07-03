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

#include "realm/upmem/upmem_proc.h"

namespace Realm {
extern Logger log_xd;
extern Logger log_taskreg;

namespace Upmem {
    extern Logger log_dpu;

    ///////////////////////////////////////////////////////////////////////
    //
    // class DPUProcessor

    DPUProcessor::DPUProcessor(DPU* _dpu, Processor _me, Realm::CoreReservationSet& crs,
        size_t _stack_size)
        : LocalTaskProcessor(_me, Processor::DPU_PROC)
        , dpu(_dpu)
        , block_on_synchronize(false)
        , ctxsync(_dpu, _dpu->device_id, crs, _dpu->module->config->cfg_max_ctxsync_threads)
    {
        Realm::CoreReservationParameters params;
        params.set_num_cores(1);
        params.set_alu_usage(params.CORE_USAGE_SHARED);
        params.set_fpu_usage(params.CORE_USAGE_SHARED);
        params.set_ldst_usage(params.CORE_USAGE_SHARED);
        params.set_max_stack_size(_stack_size); // 64 MB for each DPU

        std::string name = stringbuilder() << "DPU proc " << _me;

        // Each DPU gets a new Realm core reservation
        core_rsrv = new Realm::CoreReservation(name, crs, params);

        Realm::KernelThreadTaskScheduler* sched = new DPUTaskScheduler<Realm::KernelThreadTaskScheduler>(me, *core_rsrv, this);

        set_scheduler(sched);
    }

    DPUProcessor::~DPUProcessor(void) { delete core_rsrv; }

    bool DPUProcessor::register_task(Processor::TaskFuncID func_id,
        CodeDescriptor& codedesc,
        const ByteArrayRef& user_data)
    {
        // see if we have a function pointer to register
        const FunctionPointerImplementation* fpi = codedesc.find_impl<FunctionPointerImplementation>();

        // // if we don't have a function pointer implementation, see if we can make one
        // if(!fpi) {
        //   const std::vector<CodeTranslator *>& translators =
        //   get_runtime()->get_code_translators(); for(std::vector<CodeTranslator
        //   *>::const_iterator it = translators.begin();
        //       it != translators.end();
        //       it++)
        //     if((*it)->can_translate<FunctionPointerImplementation>(codedesc)) {
        //       FunctionPointerImplementation *newfpi =
        //       (*it)->translate<FunctionPointerImplementation>(codedesc); if(newfpi) {
        //         log_taskreg.info() << "function pointer created: trans=" << (*it)->name
        //         << " fnptr=" << (void *)(newfpi->fnptr);
        //         codedesc.add_implementation(newfpi);
        //         fpi = newfpi;
        //         break;
        //       }
        //     }
        // }

        assert(fpi != 0);

        {
            RWLock::AutoWriterLock al(task_table_mutex);

            // first, make sure we haven't seen this task id before
            if (dpu_task_table.count(func_id) > 0) {
                log_taskreg.fatal() << "duplicate task registration: proc=" << me
                                    << " func=" << func_id;
                return false;
            }

            DPUTaskTableEntry& tte = dpu_task_table[func_id];

            // figure out what type of function we have
            if (codedesc.type() == TypeConv::from_cpp_type<Processor::TaskFuncPtr>()) {
                tte.fnptr = (Processor::TaskFuncPtr)(fpi->fnptr);
                tte.stream_aware_fnptr = 0;
            } else if (codedesc.type() == TypeConv::from_cpp_type<Upmem::StreamAwareTaskFuncPtr>()) {
                tte.fnptr = 0;
                tte.stream_aware_fnptr = (Upmem::StreamAwareTaskFuncPtr)(fpi->fnptr);
            } else {
                log_taskreg.fatal() << "attempt to register a task function of improper type: "
                                    << codedesc.type();
                assert(0);
            }

            tte.user_data = user_data;
        }

        log_taskreg.info() << "task " << func_id << " registered on " << me << ": "
                           << codedesc;

        return true;
    }

    void DPUProcessor::execute_task(Processor::TaskFuncID func_id,
        const ByteArrayRef& task_args)
    {
        if (func_id == Processor::TASK_ID_PROCESSOR_NOP)
            return;

        std::map<Processor::TaskFuncID, DPUTaskTableEntry>::const_iterator it;
        {
            RWLock::AutoReaderLock al(task_table_mutex);

            it = dpu_task_table.find(func_id);
            if (it == dpu_task_table.end()) {
                log_taskreg.fatal() << "task " << func_id << " not registered on " << me;
                assert(0);
            }
        }

        const DPUTaskTableEntry& tte = it->second;

        if (tte.stream_aware_fnptr) {
            // shouldn't be here without a valid stream
            assert(ThreadLocal::current_dpu_stream);
            struct dpu_set_t* stream = ThreadLocal::current_dpu_stream->get_stream();

            log_taskreg.debug() << "task " << func_id << " executing on " << me << ": "
                                << ((void*)(tte.stream_aware_fnptr)) << " (stream aware)";

            (tte.stream_aware_fnptr)(task_args.base(), task_args.size(), tte.user_data.base(),
                tte.user_data.size(), me, stream);
        } else {
            assert(tte.fnptr);
            log_taskreg.debug() << "task " << func_id << " executing on " << me << ": "
                                << ((void*)(tte.fnptr));

            (tte.fnptr)(task_args.base(), task_args.size(), tte.user_data.base(),
                tte.user_data.size(), me);
        }
    }

    void DPUProcessor::shutdown(void)
    {
        log_dpu.info("shutting down");

        // shut down threads/scheduler
        LocalTaskProcessor::shutdown();

        ctxsync.shutdown_threads();

        CHECK_UPMEM(dpu_sync(*dpu->stream->get_stream()));
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // class DPUTaskScheduler<T>

    template <typename T>
    DPUTaskScheduler<T>::DPUTaskScheduler(Processor _proc,
        Realm::CoreReservation& _core_rsrv,
        DPUProcessor* _dpu_proc)
        : T(_proc, _core_rsrv)
        , dpu_proc(_dpu_proc)
    {
        // nothing else
    }

    template <typename T>
    DPUTaskScheduler<T>::~DPUTaskScheduler(void)
    {
    }

    template <typename T>
    bool DPUTaskScheduler<T>::execute_task(Task* task)
    {
        // use TLS to make sure that the task can find the current DPU processor
        //  UPMEM RT calls
        // TODO: either eliminate these asserts or do TLS swapping when using user threads
        assert(ThreadLocal::current_dpu_proc == 0);
        ThreadLocal::current_dpu_proc = dpu_proc;

        // bump the current stream
        // TODO: sanity-check whether this even works right when DPU tasks suspend
        assert(ThreadLocal::current_dpu_stream == 0);
        DPUStream* s = dpu_proc->dpu->get_next_task_stream();
        ThreadLocal::current_dpu_stream = s;
        assert(!ThreadLocal::created_dpu_streams);

        // we'll use a "work fence" to track when the kernels launched by this task actually
        //  finish - this must be added to the task _BEFORE_ we execute
        DPUWorkFence* fence = new DPUWorkFence(task);
        task->add_async_work_item(fence);

        // event to record the DPU start time for the task, if requested
        if (task->wants_gpu_work_start()) {
            DPUWorkStart* start = new DPUWorkStart(task);
            task->add_async_work_item(start);
            start->enqueue_on_stream(s);
        }

        bool ok = T::execute_task(task);

        // if the user could have put work on any other streams then make our
        // stream wait on those streams as well
        // TODO: update this so that it works when DPU tasks suspend
        if (ThreadLocal::created_dpu_streams) {
            s->wait_on_streams(*ThreadLocal::created_dpu_streams);
            delete ThreadLocal::created_dpu_streams;
            ThreadLocal::created_dpu_streams = 0;
        }

        // if this is our first task, we might need to decide whether
        //  full context synchronization is required for a task to be
        //  "complete"
        if (dpu_proc->dpu->module->config->cfg_task_context_sync < 0) {
            dpu_proc->dpu->module->config->cfg_task_context_sync = 1;
        }

        if ((ThreadLocal::context_sync_required > 0) || ((ThreadLocal::context_sync_required < 0) && dpu_proc->dpu->module->config->cfg_task_context_sync))
            dpu_proc->ctxsync.add_fence(fence);
        else
            fence->enqueue_on_stream(s);

            // A useful debugging macro
#ifdef FORCE_DPU_STREAM_SYNCHRONIZE
        DPU_ASSERT(dpu_sync(*s->get_stream()));
#endif

        assert(ThreadLocal::current_dpu_proc == dpu_proc);
        ThreadLocal::current_dpu_proc = 0;
        assert(ThreadLocal::current_dpu_stream == s);
        ThreadLocal::current_dpu_stream = 0;

        return ok;
    }

    template <typename T>
    void DPUTaskScheduler<T>::execute_internal_task(InternalTask* task)
    {
        // use TLS to make sure that the task can find the current DPU processor when it
        // makes
        //  UPMEM RT calls
        // TODO: either eliminate these asserts or do TLS swapping when using user threads
        assert(ThreadLocal::current_dpu_proc == 0);
        ThreadLocal::current_dpu_proc = dpu_proc;

        assert(ThreadLocal::current_dpu_stream == 0);
        DPUStream* s = dpu_proc->dpu->get_next_task_stream();
        ThreadLocal::current_dpu_stream = s;
        assert(!ThreadLocal::created_dpu_streams);

        // internal tasks aren't allowed to wait on events, so any cuda synch
        //  calls inside the call must be blocking
        dpu_proc->block_on_synchronize = true;
        // execute the internal task, whatever it is
        T::execute_internal_task(task);

        // if the user could have put work on any other streams then make our
        // stream wait on those streams as well
        // TODO: update this so that it works when DPU tasks suspend
        if (ThreadLocal::created_dpu_streams) {
            s->wait_on_streams(*ThreadLocal::created_dpu_streams);
            delete ThreadLocal::created_dpu_streams;
            ThreadLocal::created_dpu_streams = 0;
        }

        // we didn't use streams here, so synchronize the whole context
        DPU_ASSERT(dpu_sync(*s->get_stream()));
        dpu_proc->block_on_synchronize = false;

        assert(ThreadLocal::current_dpu_proc == dpu_proc);
        ThreadLocal::current_dpu_proc = 0;
        assert(ThreadLocal::current_dpu_stream == s);
        ThreadLocal::current_dpu_stream = 0;
    }

}; // namespace Upmem
}; // namespace Realm