use crate::{
    FusionBackend, FusionRuntime,
    stream::{MultiStream, OperationStreams, StreamId, execution::Operation},
};
use burn_ir::{HandleContainer, OperationIr, TensorId, TensorIr};
use burn_tensor::TensorData;

pub struct FusionServer<R: FusionRuntime> {
    streams: MultiStream<R>,
    pub(crate) handles: HandleContainer<R::FusionHandle>,
}

impl<R> FusionServer<R>
where
    R: FusionRuntime,
{
    pub fn new(device: R::FusionDevice) -> Self {
        Self {
            streams: MultiStream::new(device.clone()),
            handles: HandleContainer::new(),
        }
    }

    pub fn register(
        &mut self,
        streams: OperationStreams,
        repr: OperationIr,
        operation: Box<dyn Operation<R>>,
    ) {
        self.streams
            .register(streams, repr, operation, &mut self.handles)
    }

    pub fn drain_stream(&mut self, id: StreamId) {
        self.streams.drain(&mut self.handles, id)
    }

    pub fn create_empty_handle(&mut self) -> TensorId {
        self.handles.create_tensor_uninit()
    }

    /// Debug method to access the pre-optimized operation queue for a specific stream.
    /// Returns the raw operation sequence as written by the user.
    pub fn debug_pre_optimized(&self, stream_id: StreamId) -> Option<&Vec<OperationIr>> {
        self.streams.debug_operation_queue(stream_id)
    }

    /// Debug method to access all pre-optimized operation queues.
    /// Returns a map of stream IDs to their operation sequences.
    pub fn debug_all_pre_optimized(&self) -> std::collections::HashMap<StreamId, &Vec<OperationIr>> {
        self.streams.debug_all_operation_queues()
    }

    /// Debug method to access the post-optimized execution plans.
    /// Returns the execution plan store containing optimized strategies.
    pub(crate) fn debug_post_optimized(&self) -> &crate::stream::store::ExecutionPlanStore<R::Optimization> {
        self.streams.debug_execution_plans()
    }

    /// Debug method to get a summary of the current fusion state.
    pub fn debug_fusion_summary(&self) -> FusionDebugSummary {
        let pre_optimized = self.debug_all_pre_optimized();
        let post_optimized = self.debug_post_optimized();

        FusionDebugSummary {
            stream_count: pre_optimized.len(),
            total_operations: pre_optimized.values().map(|ops| ops.len()).sum(),
            execution_plan_count: post_optimized.debug_plan_count(),
            execution_plan_summaries: post_optimized.debug_summary(),
        }
    }

    pub fn read_float<B>(
        &mut self,
        tensor: TensorIr,
        id: StreamId,
    ) -> impl Future<Output = TensorData> + Send + use<R, B>
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        // Make sure all registered operations are executed.
        // The underlying backend can still be async.
        self.drain_stream(id);
        let tensor_float = self.handles.get_float_tensor::<B>(&tensor);
        self.streams.mark_read(id, &tensor, &self.handles);
        B::float_into_data(tensor_float)
    }

    pub fn read_int<B>(
        &mut self,
        tensor: TensorIr,
        id: StreamId,
    ) -> impl Future<Output = TensorData> + Send + use<R, B>
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        // Make sure all registered operations are executed.
        // The underlying backend can still be async.
        self.drain_stream(id);
        let tensor_int = self.handles.get_int_tensor::<B>(&tensor);
        self.streams.mark_read(id, &tensor, &self.handles);
        B::int_into_data(tensor_int)
    }

    pub fn read_bool<B>(
        &mut self,
        tensor: TensorIr,
        id: StreamId,
    ) -> impl Future<Output = TensorData> + Send + use<R, B>
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        // Make sure all registered operations are executed.
        // The underlying backend can still be async.
        self.drain_stream(id);
        let tensor_bool = self.handles.get_bool_tensor::<B>(&tensor);
        self.streams.mark_read(id, &tensor, &self.handles);
        B::bool_into_data(tensor_bool)
    }

    pub fn read_quantized<B>(
        &mut self,
        tensor: TensorIr,
        id: StreamId,
    ) -> impl Future<Output = TensorData> + Send + use<R, B>
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        // Make sure all registered operations are executed.
        // The underlying backend can still be async.
        self.drain_stream(id);
        let tensor_q = self.handles.get_quantized_tensor::<B>(&tensor);
        self.streams.mark_read(id, &tensor, &self.handles);
        B::q_into_data(tensor_q)
    }

    pub fn change_server_float<B>(
        &mut self,
        tensor: &TensorIr,
        device: &R::FusionDevice,
        server_device: &mut Self,
    ) -> TensorId
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        let tensor_float = self.handles.get_float_tensor::<B>(tensor);
        self.streams
            .mark_read(StreamId::current(), tensor, &self.handles);

        let tensor = B::float_to_device(tensor_float, device);
        let id = server_device.create_empty_handle();

        server_device
            .handles
            .register_float_tensor::<B>(&id, tensor.clone());

        id
    }

    pub fn resolve_server_float<B>(&mut self, tensor: &TensorIr) -> B::FloatTensorPrimitive
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        self.handles.get_float_tensor::<B>(tensor)
    }

    pub fn resolve_server_int<B>(&mut self, tensor: &TensorIr) -> B::IntTensorPrimitive
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        self.handles.get_int_tensor::<B>(tensor)
    }

    pub fn resolve_server_bool<B>(&mut self, tensor: &TensorIr) -> B::BoolTensorPrimitive
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        self.handles.get_bool_tensor::<B>(tensor)
    }

    pub fn change_server_int<B>(
        &mut self,
        tensor: &TensorIr,
        device: &R::FusionDevice,
        server_device: &mut Self,
    ) -> TensorId
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        let tensor_int = self.handles.get_int_tensor::<B>(tensor);
        self.streams
            .mark_read(StreamId::current(), tensor, &self.handles);
        let tensor = B::int_to_device(tensor_int, device);
        let id = server_device.create_empty_handle();

        server_device
            .handles
            .register_int_tensor::<B>(&id, tensor.clone());

        id
    }

    pub fn change_server_bool<B>(
        &mut self,
        tensor: &TensorIr,
        device: &R::FusionDevice,
        server_device: &mut Self,
    ) -> TensorId
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        let tensor_bool = self.handles.get_bool_tensor::<B>(tensor);
        self.streams
            .mark_read(StreamId::current(), tensor, &self.handles);
        let tensor = B::bool_to_device(tensor_bool, device);
        let id = server_device.create_empty_handle();

        server_device
            .handles
            .register_bool_tensor::<B>(&id, tensor.clone());

        id
    }

    pub fn change_server_quantized<B>(
        &mut self,
        tensor: &TensorIr,
        device: &R::FusionDevice,
        server_device: &mut Self,
    ) -> TensorId
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        let tensor = self.handles.get_quantized_tensor::<B>(tensor);
        let tensor = B::q_to_device(tensor, device);
        let id = server_device.create_empty_handle();

        server_device
            .handles
            .register_quantized_tensor::<B>(&id, tensor);

        id
    }
}

/// Summary information about the current fusion state for debugging.
#[derive(Debug, Clone)]
pub struct FusionDebugSummary {
    pub stream_count: usize,
    pub total_operations: usize,
    pub execution_plan_count: usize,
    pub execution_plan_summaries: Vec<crate::stream::store::ExecutionPlanSummary>,
}
