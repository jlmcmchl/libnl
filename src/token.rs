pub mod parse;

#[derive(Default, Debug, Clone, PartialEq, PartialOrd)]
pub struct Problem {
    pub headers: Headers,
    pub body: Vec<Token>, // body: &'a [u8]
}

#[derive(Default, Debug, Clone, PartialEq, PartialOrd)]
pub struct Headers {
    pub problem: ProblemStatistics,
    pub general: GeneralStatistics,
    pub nonlinear: NonlinearStatistics,
    pub network_constraints: NetworkConstraintStatistics,
    pub nonlinear_vars: NonlinearVariableStatistics,
    pub linear_network: LinearNetworkStatistics,
    pub discrete_variables: DiscreteVariableStatistics,
    pub nonzeros: NonzeroStatistics,
    pub name_lengths: MaxNameLengthStatistics,
    pub common_exprs: CommonExpressionStatistics,
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct Comment {
    body: String,
}

#[derive(Default, Debug, Clone, PartialEq, PartialOrd)]
pub struct ProblemStatistics {
    nums: Vec<u64>,
}
#[derive(Default, Debug, Clone, PartialEq, PartialOrd, Copy)]
pub struct GeneralStatistics {
    pub vars: u64,
    pub constraints: u64,
    pub objectives: u64,
    pub ranges: u64,
    pub lcons: u64,
}
#[derive(Default, Debug, Clone, PartialEq, PartialOrd, Copy)]
pub struct NonlinearStatistics {
    constraints: u64,
    objectives: u64,
}
#[derive(Default, Debug, Clone, PartialEq, PartialOrd, Copy)]
pub struct NetworkConstraintStatistics {
    nonlinear: u64,
    linear: u64,
}
#[derive(Default, Debug, Clone, PartialEq, PartialOrd, Copy)]
pub struct NonlinearVariableStatistics {
    constraints: u64,
    objectives: u64,
    both: u64,
}
#[derive(Default, Debug, Clone, PartialEq, PartialOrd, Copy)]
pub struct LinearNetworkStatistics {
    variables: u64,
    functions: u64,
    arith: u64,
    flags: u64,
}
#[derive(Default, Debug, Clone, PartialEq, PartialOrd, Copy)]
pub struct DiscreteVariableStatistics {
    binary: u64,
    integer: u64,
    nonlinear_b: u64,
    nonlinear_c: u64,
    nonlinear_o: u64,
}
#[derive(Default, Debug, Clone, PartialEq, PartialOrd, Copy)]
pub struct NonzeroStatistics {
    jacobian: u64,
    gradients: u64,
}
#[derive(Default, Debug, Clone, PartialEq, PartialOrd, Copy)]
pub struct MaxNameLengthStatistics {
    constraints: u64,
    variables: u64,
}
#[derive(Default, Debug, Clone, PartialEq, PartialOrd, Copy)]
pub struct CommonExpressionStatistics {
    b: u64,
    c: u64,
    o: u64,
    c1: u64,
    o1: u64,
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum Token {
    Number(f64),
    Variable(u64),
    Function(u64, u64),
    String(String),
    Operation(Operator),
    Segment(SegmentId),
    Single(Number),
    Pair(Number, Number),
    Trio(Number, Number, Number),
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum Number {
    Real(f64),
    Integer(i64),
}

impl Number {
    pub fn as_f64(self) -> f64 {
        match self {
            Number::Real(r) => r,
            Number::Integer(i) => i as f64,
        }
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Copy)]
pub enum Operator {
    Add,                      // { 0, "+" }
    Sub,                      // { 1, "-" }
    Mul,                      // { 2, "*" }
    Div,                      // { 3, "/" }
    Mod,                      // { 4, "mod" }
    Pow,                      // { 5, "^" }
    Less,                     // { 6, "less" }
    Min,                      // { 11, "min" } --nary
    Max,                      // { 12, "max" } --nary
    Floor,                    // { 13, "floor" }
    Ceiling,                  // { 14, "ceil" }
    AbsoluteValue,            // { 15, "abs" }
    UnaryMinus,               // { 16, "unary -" }
    Or,                       // { 20, "||" }
    And,                      // { 21, "&&" }
    LessThan,                 // { 22, "<" }
    LessThanOrEquals,         // { 23, "<=" }
    Equals,                   // { 24, "=" }
    GreatherThanOrEquals,     // { 28, ">=" }
    GreatherThan,             // { 29, ">" }
    NotEquals,                // { 30, "!=" }
    Not,                      // { 34, "!" }
    If,                       // { 35, "if" }
    HyperbolicTangent,        // { 37, "tanh" }
    Tangent,                  // { 38, "tan" }
    SquareRoot,               // { 39, "sqrt" }
    HyperbolicSine,           // { 40, "sinh" }
    Sine,                     // { 41, "sin" }
    LogarithmBase10,          // { 42, "log10" }
    NaturalLogarithm,         // { 43, "log" }
    Exponential,              // { 44, "exp" }
    HyperbolicCosine,         // { 45, "cosh" }
    Cosine,                   // { 46, "cos" }
    InverseHyperbolicTangent, // { 47, "atanh" }
    InverseTangent2,          // { 48, "atan2" }
    InverseTangent,           // { 49, "atan" }
    InverseHyperbolicSine,    // { 50, "asinh" }
    InverseSine,              // { 51, "asin" }
    InverseHyperbolicCosine,  // { 52, "acosh" }
    InverseCosine,            // { 53, "acos" }
    Sum,                      // { 54, "sum" } --nary
    TruncatedDivision,        // { 55, "div" }
    Precision,                // { 56, "precision" }
    Round,                    // { 57, "round" }
    Truncate,                 // { 58, "trunc" }
    Count,                    // { 59, "count" } --nary
    NumberOf,                 // { 60, "numberof" } --nary
    SymbolicNumberOf,         // { 61, "symbolic numberof" } --nary
    AtLeast,                  // { 62, "atleast" }
    AtMost,                   // { 63, "atmost" }
    PiecewiseLinearTerm,      // { 64, "piecewise-linear term" }
    SymbolicIf,               // { 65, "symbolic if" }
    Exactly,                  // { 66, "exactly" }
    NotAtLeast,               // { 67, "!atleast" }
    NotAtMost,                // { 68, "!atmost" }
    NotExactly,               // { 69, "!exactly" }
    ForAll,                   // { 70, "forall" } --nary
    Exists,                   // { 71, "exists" } --nary
    Implies,                  // { 72, "==>" }
    IfAndOnlyIf,              // { 73, "<==>" }
    AllDifferent,             // { 74, "alldiff" } --nary
    NotAllDifferent,          // { 75, "!alldiff" }
    PowerConstantExponent,    // { 76, "^" }
    Square,                   // { 77, "^2" }
    PowerConstantBase,        // { 78, "^" }
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum SegmentId {
    ImportedFunction(ImportedFunctionMetadata),
    Suffix(SuffixMetadata),
    VariableDefinition(VariableDefinitionMetadata),
    AlgebraicConstraint(AlgebraicConstraintMetadata),
    LogicalConstraint(LogicalConstraintMetadata),
    ObjectiveFunction(ObjectiveFunctionMetadata),
    DualInitialGuess(DualInitialGuessMetadata),
    PrimalInitialGuess(PrimalInitialGuessMetadata),
    AlgebraicConstraintBounds,
    VariableBounds,
    JacobianColumnCounts(JacobianColumnCountsMetadata),
    Jacobian(JacobianMetadata),
    Gradient(GradientMetadata),
}

pub enum ArgumentCount {
    AtLeast(u64),
    Exactly(u64),
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct ImportedFunctionMetadata {
    pub id: u64,
    pub string_arguments_allowed: bool,
    pub argument_count: i64,
    pub name: String,
}
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct SuffixMetadata {
    pub kind: SuffixKind,
    pub real: bool,
    pub count: u64,
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum SuffixKind {
    Variables,
    Constraints,
    Objectives,
    Problem,
    Unknown,
}

impl From<u8> for SuffixKind {
    fn from(value: u8) -> Self {
        match value {
            0 => SuffixKind::Variables,
            1 => SuffixKind::Constraints,
            2 => SuffixKind::Objectives,
            3 => SuffixKind::Problem,
            _ => SuffixKind::Unknown,
        }
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Copy)]
pub struct VariableDefinitionMetadata {
    pub id: u64,
    pub linear_terms: u64,
    pub constraint_info: u64,
}
#[derive(Debug, Clone, PartialEq, PartialOrd, Copy)]
pub struct AlgebraicConstraintMetadata {
    pub id: u64,
}
#[derive(Debug, Clone, PartialEq, PartialOrd, Copy)]
pub struct LogicalConstraintMetadata {
    pub id: u64,
}
#[derive(Debug, Clone, PartialEq, PartialOrd, Copy)]
pub struct ObjectiveFunctionMetadata {
    pub id: u64,
    pub typ: u8,
}
#[derive(Debug, Clone, PartialEq, PartialOrd, Copy)]
pub struct DualInitialGuessMetadata {
    pub count: u64,
}
#[derive(Debug, Clone, PartialEq, PartialOrd, Copy)]
pub struct PrimalInitialGuessMetadata {
    pub count: u64,
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Copy)]
pub struct JacobianColumnCountsMetadata {
    pub count: u64,
}
#[derive(Debug, Clone, PartialEq, PartialOrd, Copy)]
pub struct JacobianMetadata {
    pub id: u64,
    pub count: u64,
}
#[derive(Debug, Clone, PartialEq, PartialOrd, Copy)]
pub struct GradientMetadata {
    pub id: u64,
    pub count: u64,
}

impl winnow::stream::ContainsToken<Token> for Token {
    #[inline(always)]
    fn contains_token(&self, token: Token) -> bool {
        *self == token
    }
}

impl winnow::stream::ContainsToken<Token> for &'_ [Token] {
    #[inline]
    fn contains_token(&self, token: Token) -> bool {
        self.iter().any(|t| *t == token)
    }
}

impl<const LEN: usize> winnow::stream::ContainsToken<Token> for &'_ [Token; LEN] {
    #[inline]
    fn contains_token(&self, token: Token) -> bool {
        self.iter().any(|t| *t == token)
    }
}

impl<const LEN: usize> winnow::stream::ContainsToken<Token> for [Token; LEN] {
    #[inline]
    fn contains_token(&self, token: Token) -> bool {
        self.iter().any(|t| *t == token)
    }
}
