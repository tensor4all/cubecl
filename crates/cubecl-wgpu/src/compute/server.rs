use super::storage::{WgpuResource, WgpuStorage};
use crate::schedule::{BindingsResource, ScheduleTask, ScheduledWgpuBackend};
use crate::{AutoCompiler, AutoRepresentation, PrimaryMemoryMode};
use alloc::sync::Arc;
use cubecl_common::{
    backtrace::BackTrace,
    bytes::Bytes,
    profile::{ProfileDuration, TimingMethod},
    stream_id::StreamId,
};
use cubecl_core::server::{Binding, StreamErrorMode};
use cubecl_core::zspace::Shape;
use cubecl_core::{
    MemoryConfiguration, WgpuCompilationOptions,
    future::DynFut,
    prelude::*,
    server::{
        CopyDescriptor, IoError, KernelArguments, LaunchError, ProfileError, ProfilingToken,
        ResourceLimitError, ServerCommunication, ServerError, ServerUtilities,
    },
    zspace::{Strides, strides},
};
#[cfg(feature = "spirv")]
use cubecl_core::{cache::CacheOption, compilation_cache::CompilationCache, hash::StableHash};
use cubecl_ir::MemoryDeviceProperties;
use cubecl_runtime::allocator::ContiguousMemoryLayoutPolicy;
use cubecl_runtime::memory_management::{ManagedMemoryHandle, MemoryUsage};
use cubecl_runtime::{
    compiler::CubeTask,
    config::{CubeClRuntimeConfig, RuntimeConfig},
    logging::ServerLogger,
    memory_management::MemoryAllocationMode,
    server::ComputeServer,
    storage::ManagedResource,
    stream::scheduler::{SchedulerMultiStream, SchedulerMultiStreamOptions, SchedulerStrategy},
    validation::{validate_cube_dim, validate_units},
};
use hashbrown::HashMap;
use wgpu::ComputePipeline;

/// Wgpu compute server.
#[derive(Debug)]
pub struct WgpuServer {
    pub(crate) device: wgpu::Device,
    // A buffer that can be used to store stream id without extra allocations.
    streams_pool: Vec<StreamId>,
    pipelines: HashMap<KernelId, Arc<ComputePipeline>>,
    scheduler: SchedulerMultiStream<ScheduledWgpuBackend>,
    #[cfg(feature = "spirv")]
    pub(crate) spirv_cache:
        Option<CompilationCache<(u64, StableHash), cubecl_spirv::SpirvCacheEntry>>,
    pub compilation_options: WgpuCompilationOptions,
    pub(crate) backend: wgpu::Backend,
    pub(crate) utilities: Arc<ServerUtilities<Self>>,
}

/// Completion handle for one native WGPU queue submission.
#[cfg(not(target_family = "wasm"))]
#[derive(Clone, Debug)]
pub struct WgpuSubmission {
    device: wgpu::Device,
    index: wgpu::SubmissionIndex,
}

#[cfg(not(target_family = "wasm"))]
impl WgpuSubmission {
    /// Wait for this submission. Calling this method more than once is valid.
    pub fn wait(&self) -> Result<(), ServerError> {
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(self.index.clone()),
                timeout: None,
            })
            .map(|_| ())
            .map_err(|error| ServerError::Generic {
                reason: alloc::format!("WGPU submission wait failed: {error}"),
                backtrace: BackTrace::capture(),
            })
    }
}

impl ServerCommunication for WgpuServer {
    const SERVER_COMM_ENABLED: bool = false;
}

impl WgpuServer {
    /// Create a new server.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        memory_properties: MemoryDeviceProperties,
        memory_config: MemoryConfiguration,
        compilation_options: WgpuCompilationOptions,
        device: wgpu::Device,
        queue: wgpu::Queue,
        tasks_max: usize,
        primary_memory: PrimaryMemoryMode,
        backend: wgpu::Backend,
        timing_method: TimingMethod,
        utilities: ServerUtilities<Self>,
    ) -> Self {
        let backend_scheduler = ScheduledWgpuBackend::new(
            device.clone(),
            queue.clone(),
            memory_properties,
            memory_config,
            primary_memory,
            timing_method,
            tasks_max,
            utilities.logger.clone(),
        );

        let config = CubeClRuntimeConfig::get();
        let max_streams = config.streaming.max_streams;

        Self {
            compilation_options,
            streams_pool: Vec::new(),
            device,
            pipelines: HashMap::new(),
            scheduler: SchedulerMultiStream::new(
                utilities.logger.clone(),
                backend_scheduler,
                SchedulerMultiStreamOptions {
                    max_streams,
                    max_tasks: tasks_max,
                    strategy: SchedulerStrategy::Interleave,
                },
            ),
            #[cfg(feature = "spirv")]
            spirv_cache: {
                let config = cubecl_runtime::config::CubeClRuntimeConfig::get();
                if let Some(cache) = &config.compilation.cache {
                    let root = cache.root();
                    Some(CompilationCache::new(
                        "spirv",
                        CacheOption::default().name("vulkan").root(root),
                    ))
                } else {
                    None
                }
            },
            backend,
            utilities: Arc::new(utilities),
        }
    }

    /// Submit pending work for one stream and return its exact queue completion.
    #[cfg(not(target_family = "wasm"))]
    pub fn submit_stream_completion(
        &mut self,
        stream_id: StreamId,
    ) -> Result<WgpuSubmission, ServerError> {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        let device = stream.device.clone();
        let index = stream
            .flush_submission(StreamErrorMode {
                ignore: false,
                flush: true,
            })?
            .unwrap_or_else(|| stream.submission_barrier());
        Ok(WgpuSubmission { device, index })
    }

    fn prepare_bindings(
        &mut self,
        bindings: KernelArguments,
    ) -> Result<BindingsResource, ServerError> {
        // Store all the resources we'll be using. This could be eliminated if
        // there was a way to tie the lifetime of the resource to the memory handle.
        let mut resources = Vec::with_capacity(bindings.buffers.len());
        let mut reservations = Vec::with_capacity(bindings.buffers.len());

        for b in bindings.buffers.into_iter() {
            let stream = self.scheduler.stream(&b.stream);
            let resource = stream.mem_manage.get_resource(b)?;
            reservations.push(resource.acquire_gpu()?);
            resources.push(resource);
        }

        Ok(BindingsResource {
            resources,
            reservations,
            info: bindings.info,
        })
    }

    fn pipeline(
        &mut self,
        kernel: <Self as ComputeServer>::Kernel,
        bindings: &KernelArguments,
        mode: ExecutionMode,
    ) -> Result<Arc<ComputePipeline>, LaunchError> {
        let mut kernel_id = kernel.id();
        kernel_id.mode(mode);

        if let Some(pipeline) = self.pipelines.get(&kernel_id) {
            return Ok(pipeline.clone());
        }

        let cached = self.load_cached_pipeline(&kernel_id, bindings, mode)?;

        if let Some(Ok(pipeline)) = cached {
            self.pipelines.insert(kernel_id, pipeline.clone());
            return Ok(pipeline);
        }

        validate_cube_dim(&self.utilities.properties, &kernel_id)?;
        validate_units(&self.utilities.properties, &kernel_id)?;

        let mut compiler = compiler(self.backend, &self.compilation_options);
        let mut compiled = compiler.compile(self, kernel, mode)?;

        if self.scheduler.logger.compilation_activated() {
            compiled.debug_info = Some(DebugInformation::new(
                compiler.lang_tag(),
                kernel_id.clone(),
            ));
        }
        self.scheduler.logger.log_compilation(&compiled);

        self.validate_shared(&compiled.repr)?;

        // /!\ Do not delete the following commented code.
        // This is useful while working on the metal compiler.
        // Also the errors are printed nicely which is not the case when this is the runtime
        // that does it.
        // println!("SOURCE:\n{}", compiled.source);
        // {
        //     // Write shader in metal file then compile it for error
        //     std::fs::write("shader.metal", &compiled.source).expect("should write to file");
        //     let _status = std::process::Command::new("xcrun")
        //         .args(vec![
        //             "-sdk",
        //             "macosx",
        //             "metal",
        //             "-o",
        //             "shader.ir",
        //             "-c",
        //             "shader.metal",
        //             "-w",
        //         ])
        //         .status()
        //         .expect("should launch the command");
        //     // std::process::exit(status.code().unwrap());
        // }
        let repr = compiled.repr.as_ref().map(|it| it.as_ref());
        let module = self.create_module(&compiled.entrypoint_name, repr, &compiled.source, mode)?;
        let pipeline = self.create_pipeline(&compiled.entrypoint_name, repr, module, bindings);
        self.pipelines.insert(kernel_id.clone(), pipeline.clone());

        #[cfg(feature = "spirv")]
        if let Some(Err(key)) = cached
            && let Some(crate::AutoRepresentation::SpirV(kernel)) = compiled.repr
        {
            let cache = self.spirv_cache.as_mut().unwrap();
            let result = cache.insert(
                key,
                cubecl_spirv::SpirvCacheEntry::new(compiled.entrypoint_name, kernel),
            );
            if let Err(err) = result {
                log::warn!("Unable to save the SPIR-V {err:?}");
            }
        }

        Ok(pipeline)
    }

    fn validate_shared(&self, repr: &Option<crate::AutoRepresentation>) -> Result<(), LaunchError> {
        let shared_bytes = repr.as_ref().map(|repr| match repr {
            AutoRepresentation::Wgsl(repr) => repr.shared_memory_bytes(),
            #[cfg(feature = "msl")]
            AutoRepresentation::Msl(repr) => repr.shared_memory_size(),
            #[cfg(feature = "spirv")]
            AutoRepresentation::SpirV(repr) => repr.shared_size,
        });
        let max_smem = self.utilities.properties.hardware.max_shared_memory_size;
        if let Some(shared_bytes) = shared_bytes
            && shared_bytes > max_smem
        {
            Err(ResourceLimitError::SharedMemory {
                requested: shared_bytes,
                max: max_smem,
                backtrace: BackTrace::capture(),
            }
            .into())
        } else {
            Ok(())
        }
    }
}

impl ComputeServer for WgpuServer {
    type Kernel = Box<dyn CubeTask<AutoCompiler>>;
    type Storage = WgpuStorage;
    type MemoryLayoutPolicy = ContiguousMemoryLayoutPolicy;
    type Info = wgpu::Backend;

    fn logger(&self) -> Arc<ServerLogger> {
        self.scheduler.logger.clone()
    }

    fn utilities(&self) -> Arc<ServerUtilities<Self>> {
        self.utilities.clone()
    }

    fn staging(
        &mut self,
        _sizes: &[usize],
        _stream_id: StreamId,
    ) -> Result<Vec<Bytes>, ServerError> {
        // TODO: Check if using a staging buffer is useful here.
        Err(IoError::UnsupportedIoOperation {
            backtrace: BackTrace::capture(),
        }
        .into())
    }

    fn initialize_memory(&mut self, memory: ManagedMemoryHandle, size: u64, stream_id: StreamId) {
        let stream = self.scheduler.stream(&stream_id);
        let reserved = stream.empty(size).unwrap();
        stream.mem_manage.bind(reserved, memory);
    }

    fn read(
        &mut self,
        descriptors: Vec<CopyDescriptor>,
        stream_id: StreamId,
    ) -> DynFut<Result<Vec<Bytes>, ServerError>> {
        let mut streams = vec![stream_id];
        let mut resources = Vec::with_capacity(descriptors.len());
        for desc in descriptors {
            if contiguous_strides(&desc.shape) != desc.strides {
                return Box::pin(async {
                    Err(IoError::UnsupportedStrides {
                        backtrace: BackTrace::capture(),
                    }
                    .into())
                });
            }
            if !streams.contains(&desc.handle.stream) {
                streams.push(desc.handle.stream);
            }
            let stream = self.scheduler.stream(&desc.handle.stream);
            let resource = match stream.mem_manage.get_resource(desc.handle) {
                Ok(val) => val,
                Err(err) => return Box::pin(async move { Err(err.into()) }),
            };
            if let Err(err) = resource.ensure_gpu_access() {
                return Box::pin(async move { Err(err.into()) });
            }
            resources.push((resource, desc.shape, desc.elem_size));
        }

        self.scheduler.execute_streams(streams);

        let stream = self.scheduler.stream(&stream_id);
        stream.read_resources(resources)
    }

    unsafe fn write(&mut self, descriptors: Vec<(CopyDescriptor, Bytes)>, stream_id: StreamId) {
        for (desc, data) in descriptors {
            let stream = self.scheduler.stream(&desc.handle.stream);

            if contiguous_strides(&desc.shape) != desc.strides {
                stream.error(ServerError::Io(IoError::UnsupportedStrides {
                    backtrace: BackTrace::capture(),
                }));
                return;
            }

            let resource = match stream.mem_manage.get_resource(desc.handle) {
                Ok(r) => r,
                Err(err) => {
                    stream.error(ServerError::Io(err));
                    return;
                }
            };
            let reservation = match resource.acquire_gpu() {
                Ok(reservation) => reservation,
                Err(err) => {
                    stream.error(err.into());
                    return;
                }
            };
            let task = ScheduleTask::Write {
                data,
                buffer: resource,
                reservation,
            };

            self.scheduler.register(stream_id, task, &[]);
        }
    }

    fn get_resource(
        &mut self,
        binding: Binding,
        stream_id: StreamId,
    ) -> Result<ManagedResource<WgpuResource>, ServerError> {
        let mut streams = vec![stream_id];
        if binding.stream != stream_id {
            streams.push(binding.stream);
        }
        self.scheduler.execute_streams(streams);
        let stream = self.scheduler.stream(&binding.stream);
        let memory = binding.memory.clone();
        let resource = stream.mem_manage.get_resource(binding)?;

        Ok(ManagedResource::new(memory, resource))
    }

    unsafe fn launch(
        &mut self,
        kernel: Self::Kernel,
        count: CubeCount,
        args: KernelArguments,
        mode: ExecutionMode,
        stream_id: StreamId,
    ) {
        let pipeline = match self.pipeline(kernel, &args, mode) {
            Ok(val) => val,
            Err(err) => {
                // We make the stream that would execute the kernel in error.
                let stream = self.scheduler.stream(&stream_id);
                stream.errors.push(ServerError::Launch(err));
                return;
            }
        };

        self.streams_pool.clear();
        args.buffers
            .iter()
            .for_each(|b| self.streams_pool.push(b.stream));

        let resources = match self.prepare_bindings(args) {
            Ok(val) => val,
            Err(err) => {
                // We make the stream that would execute the kernel in error.
                let stream = self.scheduler.stream(&stream_id);
                stream.errors.push(err);
                return;
            }
        };
        let task = ScheduleTask::Execute {
            pipeline,
            count,
            resources,
        };

        self.scheduler.register(stream_id, task, &self.streams_pool);
    }

    fn flush(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        self.scheduler.execute_streams(vec![stream_id]);

        let stream = self.scheduler.stream(&stream_id);

        stream.flush(StreamErrorMode {
            ignore: false,
            flush: true,
        })
    }

    /// Returns the total time of GPU work this sync completes.
    fn sync(&mut self, stream_id: StreamId) -> DynFut<Result<(), ServerError>> {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);

        stream.sync()
    }

    fn start_profile(&mut self, stream_id: StreamId) -> Result<ProfilingToken, ServerError> {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        stream.start_profile()
    }

    fn end_profile(
        &mut self,
        stream_id: StreamId,
        token: ProfilingToken,
    ) -> Result<ProfileDuration, ProfileError> {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);

        stream.end_profile(token)
    }

    fn memory_usage(&mut self, stream_id: StreamId) -> Result<MemoryUsage, ServerError> {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        Ok(stream.mem_manage.memory_usage())
    }

    fn memory_cleanup(&mut self, stream_id: StreamId) {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        stream.mem_manage.memory_cleanup(true);
    }

    fn allocation_mode(&mut self, mode: MemoryAllocationMode, stream_id: StreamId) {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        stream.mem_manage.mode(mode);
    }
}

fn compiler(backend: wgpu::Backend, options: &WgpuCompilationOptions) -> AutoCompiler {
    let _ = options; // Unused without `spirv` feature
    match backend {
        #[cfg(feature = "spirv")]
        wgpu::Backend::Vulkan if options.supports_vulkan => AutoCompiler::SpirV(Default::default()),
        #[cfg(feature = "msl")]
        wgpu::Backend::Metal => AutoCompiler::Msl(Default::default()),
        _ => AutoCompiler::Wgsl(Default::default()),
    }
}

pub(crate) fn contiguous_strides(shape: &Shape) -> Strides {
    let rank = shape.len();
    let mut strides = strides![1; rank];
    for i in (0..rank - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use crate::{HostAccessError, RuntimeOptions, WgpuDevice, runtime};
    use cubecl_common::future;
    use cubecl_runtime::server::Handle;

    #[test]
    fn scheduled_write_reserves_gpu_access_until_queue_submission() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::METAL,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        if future::block_on(instance.enumerate_adapters(wgpu::Backends::METAL)).is_empty() {
            eprintln!("skipping scheduled-write reservation test: no Metal adapter is available");
            return;
        }

        let options = RuntimeOptions {
            tasks_max: 32,
            memory_config: MemoryConfiguration::ExclusivePages,
            primary_memory: PrimaryMemoryMode::HostVisible,
        };
        let setup = future::block_on(runtime::create_setup_for_device(
            &WgpuDevice::DefaultDevice,
            wgpu::Backend::Metal,
            options.primary_memory,
        ));
        let mut server = runtime::create_server(setup, options);
        let stream_id = StreamId::current();
        let handle = Handle::new(stream_id, 16);
        server.initialize_memory(handle.memory.clone(), 16, stream_id);
        let managed = server
            .get_resource(handle.clone().binding(), stream_id)
            .unwrap();
        let resource = managed.resource().clone();

        // SAFETY: the descriptor's binding belongs to the allocation created above, so this test owns
        // the destination storage.
        unsafe {
            server.write(
                vec![(
                    CopyDescriptor::new(handle.binding(), [16].into(), [1].into(), 1),
                    Bytes::from_bytes_vec(vec![7; 16]),
                )],
                stream_id,
            )
        };

        let host_attempt = std::thread::spawn(move || resource.map_read().unwrap_err());
        assert_eq!(
            host_attempt.join().unwrap(),
            HostAccessError::GpuAccessInProgress
        );

        server.flush(stream_id).unwrap();
        let guard = managed.resource().map_read().unwrap();
        assert_eq!(&*guard, &[7; 16]);
    }
}

#[cfg(all(test, not(target_family = "wasm")))]
mod submission_completion_tests {
    use super::*;

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn submission_completion_api_is_thread_safe() {
        assert_send_sync::<WgpuSubmission>();

        let _submit: fn(&mut WgpuServer, StreamId) -> Result<WgpuSubmission, ServerError> =
            WgpuServer::submit_stream_completion;
    }

    #[test]
    #[ignore = "requires a Vulkan adapter"]
    fn stream_submission_completion_can_be_waited_repeatedly() {
        use crate::{PrimaryMemoryMode, RuntimeOptions, WgpuDevice, runtime};
        use cubecl_common::future;
        use cubecl_runtime::server::ComputeServer;

        let options = RuntimeOptions {
            tasks_max: 32,
            memory_config: MemoryConfiguration::ExclusivePages,
            primary_memory: PrimaryMemoryMode::DeviceLocal,
        };
        let setup = future::block_on(runtime::create_setup_for_device(
            &WgpuDevice::DefaultDevice,
            wgpu::Backend::Vulkan,
            options.primary_memory,
        ));
        let mut server = runtime::create_server(setup, options);
        let stream_id = StreamId::current();
        let _resource = server
            .scheduler
            .stream(&stream_id)
            .create_uniform(&[1, 2, 3, 4]);

        let completion = server.submit_stream_completion(stream_id).unwrap();
        completion.wait().unwrap();
        completion.wait().unwrap();
        drop(completion);
        drop(_resource);
        future::block_on(server.sync(stream_id)).unwrap();
    }
}
