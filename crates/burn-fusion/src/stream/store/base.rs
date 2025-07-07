use std::sync::Arc;

use crate::search::BlockOptimization;

use super::{ExecutionPlanIndex, InsertQuery, SearchQuery};
use burn_ir::OperationIr;
use serde::{Deserialize, Serialize};

/// The store that contains all explorations done on a device.
#[derive(Default)]
pub(crate) struct ExecutionPlanStore<O> {
    plans: Vec<ExecutionPlan<O>>,
    index: ExecutionPlanIndex,
}

/// How a list of operations should be executed.
#[derive(PartialEq, Debug, Clone)]
pub(crate) enum ExecutionStrategy<O> {
    /// An optimization was found, and therefore should be executed.
    Optimization { opt: O, ordering: Arc<Vec<usize>> },
    /// No optimization was found, each operation should be executed individually.
    Operations { ordering: Arc<Vec<usize>> },
    /// A composition of multiple execution strategies.
    Composed(Vec<Box<Self>>),
}

/// The trigger that indicates when to stop exploring.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub(crate) enum ExecutionTrigger {
    OnOperations(Vec<OperationIr>),
    OnSync,
    Always,
}

/// The unique identifier for an exploration that was executed.
pub(crate) type ExecutionPlanId = usize;

/// The outcome of an exploration that can be stored.
#[derive(Debug)]
pub(crate) struct ExecutionPlan<O> {
    /// The operations on which the exploration is related to.
    pub(crate) operations: Vec<OperationIr>,
    /// The criteria that signal when this plan should be executed. Only one trigger is necessary.
    pub(crate) triggers: Vec<ExecutionTrigger>,
    /// The optimization that should be used when executing this plan.
    pub(crate) optimization: BlockOptimization<O>,
}

impl<O> ExecutionPlanStore<O> {
    pub fn new() -> Self {
        Self {
            plans: Vec::new(),
            index: ExecutionPlanIndex::default(),
        }
    }

    pub fn find(&self, query: SearchQuery<'_>) -> Vec<ExecutionPlanId> {
        self.index.find(query)
    }

    pub fn add(&mut self, exploration: ExecutionPlan<O>) -> ExecutionPlanId {
        if exploration.operations.is_empty() {
            panic!("Can't add an empty optimization.");
        }

        let id = self.plans.len();

        self.index.insert(InsertQuery::NewPlan {
            operations: &exploration.operations,
            id,
        });

        self.plans.push(exploration);

        id
    }

    pub fn get_mut_unchecked(&mut self, id: ExecutionPlanId) -> &mut ExecutionPlan<O> {
        &mut self.plans[id]
    }

    pub fn get_unchecked(&self, id: ExecutionPlanId) -> &ExecutionPlan<O> {
        &self.plans[id]
    }

    /// Add a new end condition for an optimization.
    pub fn add_trigger(&mut self, id: ExecutionPlanId, trigger: ExecutionTrigger) {
        let criteria = &mut self.plans[id].triggers;

        if !criteria.contains(&trigger) {
            criteria.push(trigger);
        }
    }

    #[allow(dead_code)]
    /// Debug method to access all execution plans.
    /// Returns the post-optimized execution plans.
    pub fn debug_plans(&self) -> &Vec<ExecutionPlan<O>> {
        &self.plans
    }

    /// Debug method to get the number of execution plans.
    pub fn debug_plan_count(&self) -> usize {
        self.plans.len()
    }

    /// Debug method to serialize execution plans to JSON.
    /// Note: This requires the optimization type O and related types to implement Serialize.
    /// Currently disabled due to missing Serialize implementations.
    #[allow(dead_code)]
    pub fn debug_to_json(&self) -> Result<String, serde_json::Error>
    where
        O: serde::Serialize,
    {
        // TODO: Enable when ExecutionPlan, BlockOptimization, and ExecutionTrigger implement Serialize
        // For now, return a simple error message as JSON
        Ok("{\"error\": \"Serialization not yet implemented for ExecutionPlan\"}".to_string())
    }

    /// Debug method to get a summary of all execution plans.
    pub fn debug_summary(&self) -> Vec<ExecutionPlanSummary> {
        self.plans
            .iter()
            .enumerate()
            .map(|(id, plan)| ExecutionPlanSummary {
                id,
                operation_count: plan.operations.len(),
                trigger_count: plan.triggers.len(),
            })
            .collect()
    }

    /// Debug method to get execution plan summaries with operation types.
    pub fn debug_summary_with_operations(&self) -> Vec<ExecutionPlanSummaryWithOps> {
        self.plans
            .iter()
            .enumerate()
            .map(|(id, plan)| {
                let operation_types: Vec<String> = plan.operations
                    .iter()
                    .map(|op| crate::debug::operation_type_name(op))
                    .collect();

                ExecutionPlanSummaryWithOps {
                    id,
                    operation_count: plan.operations.len(),
                    trigger_count: plan.triggers.len(),
                    operation_types,
                }
            })
            .collect()
    }

    /// Debug method to get detailed execution plan information.
    pub fn debug_detailed_plans(&self) -> Vec<ExecutionPlanDetails>
    where
        O: std::fmt::Debug,
    {
        self.plans.iter().enumerate().map(|(id, plan)| {
            ExecutionPlanDetails {
                id,
                operation_count: plan.operations.len(),
                operations: plan.operations.iter().map(|op| format!("{:?}", op)).collect(),
                trigger_count: plan.triggers.len(),
                triggers: plan.triggers.iter().map(|trigger| format!("{:?}", trigger)).collect(),
                optimization_info: format!("{:?}", plan.optimization),
            }
        }).collect()
    }

    /// Debug method to access the actual optimization objects.
    /// This allows access to FuseTrace for CubeCL fusion backends.
    pub fn debug_optimizations(&self) -> Vec<ExecutionPlanOptimization<'_, O>> {
        self.plans.iter().enumerate().map(|(id, plan)| {
            ExecutionPlanOptimization {
                id,
                operation_count: plan.operations.len(),
                operations: plan.operations.clone(),
                optimization_strategy: &plan.optimization.strategy,
            }
        }).collect()
    }

    /// Debug method to extract FuseTrace information from execution plans.
    pub fn debug_fuse_trace_info(&self) -> Vec<String>
    where
        O: std::fmt::Debug,
    {
        self.plans.iter().map(|plan| {
            crate::debug::extract_fuse_trace_info(&plan.optimization.strategy)
        }).collect()
    }
}

/// Summary information about an execution plan for debugging.
#[derive(Debug, Clone)]
pub struct ExecutionPlanSummary {
    pub id: usize,
    pub operation_count: usize,
    pub trigger_count: usize,
}

/// Summary information about an execution plan with operation types for debugging.
#[derive(Debug, Clone)]
pub struct ExecutionPlanSummaryWithOps {
    pub id: usize,
    pub operation_count: usize,
    pub trigger_count: usize,
    pub operation_types: Vec<String>,
}

/// Detailed information about an execution plan for debugging.
#[derive(Debug, Clone)]
pub struct ExecutionPlanDetails {
    pub id: usize,
    pub operation_count: usize,
    pub operations: Vec<String>,
    pub trigger_count: usize,
    pub triggers: Vec<String>,
    pub optimization_info: String,
}

/// Execution plan with access to the actual optimization object.
/// This allows access to FuseTrace for CubeCL fusion backends.
#[derive(Debug)]
pub struct ExecutionPlanOptimization<'a, O> {
    pub id: usize,
    pub operation_count: usize,
    pub operations: Vec<OperationIr>,
    pub optimization_strategy: &'a ExecutionStrategy<O>,
}
