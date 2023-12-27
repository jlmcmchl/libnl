use crate::token::{
    AlgebraicConstraintMetadata, DualInitialGuessMetadata, GradientMetadata,
    ImportedFunctionMetadata, JacobianColumnCountsMetadata, JacobianMetadata,
    LogicalConstraintMetadata, ObjectiveFunctionMetadata, Operator, PrimalInitialGuessMetadata,
    SuffixMetadata, VariableDefinitionMetadata,
};

pub mod parse;

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct Problem {
    pub segments: Vec<Segment>,
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum Segment {
    ImportedFunction(ImportedFunctionMetadata),
    Suffix(SuffixMetadata, Suffix),
    VariableDefinition(VariableDefinitionMetadata, VariableDefinition),
    AlgebraicConstraint(AlgebraicConstraintMetadata, ExpressionGraph),
    LogicalConstraint(LogicalConstraintMetadata, ExpressionGraph),
    ObjectiveFunction(ObjectiveFunctionMetadata, ExpressionGraph),
    DualInitialGuess(DualInitialGuessMetadata, Vec<(u64, f64)>),
    PrimalInitialGuess(PrimalInitialGuessMetadata, Vec<(u64, f64)>),
    AlgebraicConstraintBounds(Vec<Constraint>),
    VariableBounds(Vec<Constraint>),
    JacobianColumnCounts(JacobianColumnCountsMetadata, Vec<i64>),
    Jacobian(JacobianMetadata, Vec<(u64, f64)>),
    Gradient(GradientMetadata, Vec<(u64, f64)>),
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum Suffix {
    Real(Vec<(u64, f64)>),
    Integer(Vec<(u64, i64)>),
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct VariableDefinition {
    sum_terms: Vec<(u64, f64)>,
    expression: ExpressionGraph,
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum ExpressionGraph {
    UnaryOperator(Operator, Box<ExpressionGraph>),
    BinaryOperator(Operator, Box<(ExpressionGraph, ExpressionGraph)>),
    NaryOperator(Operator, Vec<ExpressionGraph>),
    IFThenElseOperator(
        Operator,
        Box<(ExpressionGraph, ExpressionGraph, ExpressionGraph)>,
    ),
    Number(f64),
    Variable(u64),
    Function(u64, Vec<ExpressionGraph>),
    String(String),
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum Constraint {
    Between(f64, f64),
    LessThan(f64),
    GreaterThan(f64),
    Unconstrained,
    Equal(f64),
    Complementary(Mode, u64),
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum Mode {
    Unbounded,
    FiniteLowerBound,
    FiniteUpperBound,
    FullyBounded,
}
