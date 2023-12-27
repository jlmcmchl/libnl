use winnow::{
    binary::length_repeat,
    combinator::{cond, fail, peek, repeat, success, todo},
    dispatch,
    error::InputError,
    seq,
    token::any,
    Parser, Stateful, trace::trace,
};

use crate::token::{
    AlgebraicConstraintMetadata, DualInitialGuessMetadata, GradientMetadata, Headers,
    ImportedFunctionMetadata, JacobianColumnCountsMetadata, JacobianMetadata,
    LogicalConstraintMetadata, Number, ObjectiveFunctionMetadata, Operator,
    PrimalInitialGuessMetadata, SegmentId, SuffixMetadata, Token, VariableDefinitionMetadata,
};

use super::{Constraint, ExpressionGraph, Mode, Problem, Segment, Suffix};

type Stream<'a> = Stateful<&'a [Token], &'a Headers>;

type Error<'a> = InputError<Stream<'a>>;

#[allow(unused)]
fn problem<'a>() -> impl Parser<Stream<'a>, Problem, Error<'a>> {
    seq! {
        Problem {
            segments: repeat(0..,
                trace("segment", segment())
            ),
        }
    }
}

fn segment<'a>() -> impl Parser<Stream<'a>, Segment, Error<'a>> {
    dispatch! {any;
        Token::Segment(SegmentId::ImportedFunction(metadata)) => trace("imported_function", imported_function(metadata)),
        Token::Segment(SegmentId::Suffix(metadata)) => trace("suffix", suffix(metadata)),
        Token::Segment(SegmentId::VariableDefinition(metadata)) => trace("variable_definition", variable_definition(metadata)),
        Token::Segment(SegmentId::AlgebraicConstraint(metadata)) => trace("algebraic_constraint", algebraic_constraint(metadata)),
        Token::Segment(SegmentId::LogicalConstraint(metadata)) => trace("logical_constraint", logical_constraint(metadata)),
        Token::Segment(SegmentId::ObjectiveFunction(metadata)) => trace("objective_function", objective_function(metadata)),
        Token::Segment(SegmentId::DualInitialGuess(metadata)) => trace("dual_initial_guess", dual_initial_guess(metadata)),
        Token::Segment(SegmentId::PrimalInitialGuess(metadata)) => trace("primal_initial_guess", primal_initial_guess(metadata)),
        Token::Segment(SegmentId::AlgebraicConstraintBounds) => trace("algebraic_constraint_bounds", algebraic_constraint_bounds()),
        Token::Segment(SegmentId::VariableBounds) => trace("variable_bounds", variable_bounds()),
        Token::Segment(SegmentId::JacobianColumnCounts(metadata)) => trace("jacobian_column_counts", jacobian_column_counts(metadata)),
        Token::Segment(SegmentId::Jacobian(metadata)) => trace("jacobian", jacobian(metadata)),
        Token::Segment(SegmentId::Gradient(metadata)) => trace("gradient", gradient(metadata)),
        _ =>  fail,
    }
}

fn imported_function<'a>(
    metadata: ImportedFunctionMetadata,
) -> impl Parser<Stream<'a>, Segment, Error<'a>> {
    success(Segment::ImportedFunction(metadata))
}

fn suffix<'a>(metadata: SuffixMetadata) -> impl Parser<Stream<'a>, Segment, Error<'a>> {
    let len = metadata.count as usize;

    (
        cond(metadata.real, trace("suffix_real", suffix_real(len))),
        cond(!metadata.real, trace("suffix_integer", suffix_integer(len))),
    )
        .map(move |(real, int)| Segment::Suffix(metadata.clone(), int.or(real).unwrap()))
}

fn suffix_real<'a>(len: usize) -> impl Parser<Stream<'a>, Suffix, Error<'a>> {
    repeat(len, dispatch!{peek(any);
        Token::Pair(Number::Integer(first), second) => success((first as u64, second.as_f64())),
    _ => fail
})
    .map(Suffix::Real)
}

fn suffix_integer<'a>(len: usize) -> impl Parser<Stream<'a>, Suffix, Error<'a>> {
    repeat(len, dispatch!{peek(any);
        Token::Pair(Number::Integer(first), Number::Integer(second)) => success((first as u64, second)),
    _ => fail
})
    .map(Suffix::Integer)
}

fn variable_definition<'a>(
    metadata: VariableDefinitionMetadata,
) -> impl Parser<Stream<'a>, Segment, Error<'a>> {
    (
        repeat(
            metadata.linear_terms as usize,
            dispatch! {
                any;
                Token::Pair(Number::Integer(id), scalar) => success((id as u64, scalar.as_f64())),
                _ => fail
            },
        ),
        expression_graph(),
    )
        .map(move |(sum_terms, expression)| {
            Segment::VariableDefinition(
                metadata,
                super::VariableDefinition {
                    sum_terms,
                    expression,
                },
            )
        })
}

fn algebraic_constraint<'a>(
    metadata: AlgebraicConstraintMetadata,
) -> impl Parser<Stream<'a>, Segment, Error<'a>> {
    expression_graph().map(move |expression| Segment::AlgebraicConstraint(metadata, expression))
}

fn logical_constraint<'a>(
    metadata: LogicalConstraintMetadata,
) -> impl Parser<Stream<'a>, Segment, Error<'a>> {
    expression_graph().map(move |expression| Segment::LogicalConstraint(metadata, expression))
}

fn objective_function<'a>(
    metadata: ObjectiveFunctionMetadata,
) -> impl Parser<Stream<'a>, Segment, Error<'a>> {
    expression_graph().map(move |expression| Segment::ObjectiveFunction(metadata, expression))
}

fn dual_initial_guess<'a>(
    metadata: DualInitialGuessMetadata,
) -> impl Parser<Stream<'a>, Segment, Error<'a>> {
    repeat(
        metadata.count as usize,
        dispatch! {
            any;
            Token::Pair(Number::Integer(id), scalar) => success((id as u64, scalar.as_f64())),
            _ => fail
        },
    )
    .map(move |guesses| Segment::DualInitialGuess(metadata, guesses))
}

fn primal_initial_guess<'a>(
    metadata: PrimalInitialGuessMetadata,
) -> impl Parser<Stream<'a>, Segment, Error<'a>> {
    repeat(
        metadata.count as usize,
        dispatch! {
            any;
            Token::Pair(Number::Integer(id), scalar) => success((id as u64, scalar.as_f64())),
            _ => fail
        },
    )
    .map(move |guesses| Segment::PrimalInitialGuess(metadata, guesses))
}

fn algebraic_constraint_bounds<'a>() -> impl Parser<Stream<'a>, Segment, Error<'a>> {
    |input: &mut Stateful<&'a [Token], &'a Headers>| {
        repeat(input.state.general.constraints as usize, dispatch! {
            any;
            Token::Trio(Number::Integer(0), lower, upper) => success(Constraint::Between(lower.as_f64(), upper.as_f64())),
            Token::Pair(Number::Integer(1), lower) => success(Constraint::LessThan(lower.as_f64())),
            Token::Pair(Number::Integer(2), upper) => success(Constraint::GreaterThan(upper.as_f64())),
            Token::Single(Number::Integer(3)) => success(Constraint::Unconstrained),
            Token::Pair(Number::Integer(4), equal) => success(Constraint::Equal(equal.as_f64())),
            Token::Trio(Number::Integer(5), Number::Integer(0), Number::Integer(id)) => success(Constraint::Complementary(Mode::Unbounded, id as u64)),
            Token::Trio(Number::Integer(5), Number::Integer(1), Number::Integer(id)) => success(Constraint::Complementary(Mode::FiniteLowerBound, id as u64)),
            Token::Trio(Number::Integer(5), Number::Integer(2), Number::Integer(id)) => success(Constraint::Complementary(Mode::FiniteUpperBound, id as u64)),
            Token::Trio(Number::Integer(5), Number::Integer(3), Number::Integer(id)) => success(Constraint::Complementary(Mode::FullyBounded, id as u64)),
            _ => fail
        }).map(Segment::AlgebraicConstraintBounds)
        .parse_next(input)
    }
}

fn variable_bounds<'a>() -> impl Parser<Stream<'a>, Segment, Error<'a>> {
    |input: &mut Stateful<&'a [Token], &'a Headers>| {
        repeat(input.state.general.vars as usize, dispatch! {
            any;
            Token::Trio(Number::Integer(0), lower, upper) => success(Constraint::Between(lower.as_f64(), upper.as_f64())),
            Token::Pair(Number::Integer(1), lower) => success(Constraint::LessThan(lower.as_f64())),
            Token::Pair(Number::Integer(2), upper) => success(Constraint::GreaterThan(upper.as_f64())),
            Token::Single(Number::Integer(3)) => success(Constraint::Unconstrained),
            Token::Pair(Number::Integer(4), equal) => success(Constraint::Equal(equal.as_f64())),
            _ => fail
        }).map(Segment::VariableBounds)
        .parse_next(input)
    }
}

fn jacobian_column_counts<'a>(
    metadata: JacobianColumnCountsMetadata,
) -> impl Parser<Stream<'a>, Segment, Error<'a>> {
    repeat(
        metadata.count as usize,
        dispatch! {any;
            Token::Single(Number::Integer(val)) => success(val),
            _ => fail
        },
    )
    .map(move |cums| Segment::JacobianColumnCounts(metadata, cums))
}

fn jacobian<'a>(metadata: JacobianMetadata) -> impl Parser<Stream<'a>, Segment, Error<'a>> {
    repeat(
        metadata.count as usize,
        dispatch! {any;
            Token::Pair(Number::Integer(val), scalar) => success((val as u64, scalar.as_f64())),
            _ => fail
        },
    )
    .map(move |cums| Segment::Jacobian(metadata, cums))
}

fn gradient<'a>(metadata: GradientMetadata) -> impl Parser<Stream<'a>, Segment, Error<'a>> {
    repeat(
        metadata.count as usize,
        dispatch! {any;
            Token::Pair(Number::Integer(val), scalar) => success((val as u64, scalar.as_f64())),
            _ => fail
        },
    )
    .map(move |cums| Segment::Gradient(metadata, cums))
}

fn expression_graph<'a>() -> impl Parser<Stream<'a>, ExpressionGraph, Error<'a>> {
    trace("expression_graph", dispatch! {any;
        Token::Number(val) => trace("number", success(ExpressionGraph::Number(val))),
        Token::Variable(id) => trace("variable", success(ExpressionGraph::Variable(id))),
        Token::Function(id, args) => trace("function", repeat(args as usize, expression_graph()).map(|args| ExpressionGraph::Function(id, args))),
        Token::String(text) => trace("string", success(ExpressionGraph::String(text))),
        Token::Operation(op) => dispatch! {success(op);
            // Unary
            Operator::Floor |
            Operator::Ceiling |
            Operator::AbsoluteValue |
            Operator::UnaryMinus |
            Operator::Not |
            Operator::HyperbolicTangent |
            Operator::Tangent |
            Operator::SquareRoot |
            Operator::HyperbolicSine |
            Operator::Sine |
            Operator::LogarithmBase10 |
            Operator::NaturalLogarithm |
            Operator::Exponential |
            Operator::HyperbolicCosine |
            Operator::Cosine |
            Operator::InverseHyperbolicTangent |
            Operator::InverseTangent |
            Operator::InverseHyperbolicSine |
            Operator::InverseSine |
            Operator::InverseHyperbolicCosine |
            Operator::InverseCosine => trace("unary op", expression_graph().map(|expression| ExpressionGraph::UnaryOperator(op, Box::new(expression)))),

            // Binary
            Operator::Add |
            Operator::Sub |
            Operator::Mul |
            Operator::Div |
            Operator::Mod |
            Operator::Pow |
            Operator::Less |
            Operator::Or |
            Operator::And |
            Operator::LessThan |
            Operator::LessThanOrEquals |
            Operator::Equals |
            Operator::GreatherThanOrEquals |
            Operator::GreatherThan |
            Operator::NotEquals |
            Operator::InverseTangent2 |
            Operator::TruncatedDivision |
            Operator::Precision |
            Operator::Round |
            Operator::Truncate |
            Operator::IfAndOnlyIf => trace("binary_op", (expression_graph(), expression_graph()).map(|exprs| ExpressionGraph::BinaryOperator(op, Box::new(exprs)))),

            // N-ary
            Operator::Min |
            Operator::Max |
            Operator::Sum |
            Operator::Count |
            Operator::NumberOf |
            Operator::SymbolicNumberOf |
            Operator::ForAll |
            Operator::Exists |
            Operator::AllDifferent => trace("nary op", length_repeat(single().map(|val| val as usize), expression_graph()).map(|exprs| ExpressionGraph::NaryOperator(op, exprs))),

            // if-then-else
            Operator::If |
            Operator::SymbolicIf |
            Operator::Implies => trace("if-then-else op", (expression_graph(), expression_graph(), expression_graph()).map(|exprs| ExpressionGraph::IFThenElseOperator(op, Box::new(exprs)))),

            // Undetermined
            Operator::AtLeast |
            Operator::AtMost |
            Operator::PiecewiseLinearTerm |
            Operator::Exactly |
            Operator::NotAtLeast |
            Operator::NotAtMost |
            Operator::NotExactly |
            Operator::NotAllDifferent |
            Operator::PowerConstantExponent |
            Operator::Square |
            Operator::PowerConstantBase => todo
        },
        _ => fail
    })
}

fn single<'a>() -> impl Parser<Stream<'a>, f64, Error<'a>> {
    dispatch! { any;
        Token::Single(v) => success(v.as_f64()),
        _ => fail
    }
}

mod test {
    #[allow(unused_imports)]
    use std::{fs::File, io::Read};

    #[allow(unused_imports)]
    use winnow::{Stateful, Parser};

    #[allow(unused_imports)]
    use crate::token::Token;

    #[allow(unused_imports)]
    use super::problem;

    #[test]
    fn parses_all_samples() {
        for entry in glob::glob("data/*.nl").expect("Failed to read glob pattern") {
            match entry {
                Ok(path) => {
                    let mut file = File::open(&path)
                        .unwrap_or_else(|_| panic!("Could not open file {:?}", path.display()));
                    let mut content = String::new();
                    file.read_to_string(&mut content)
                        .unwrap_or_else(|_| panic!("Failed to read file {:?}", path.display()));

                    match crate::token::parse::problem(&mut content.as_str()) {
                        Ok(intermediate) => {
                            let input = Stateful::<&[Token], _> {
                                input: &intermediate.body,
                                state: &intermediate.headers,
                            };
                            match problem().parse(input) {
                                Ok(_p) => {},
                                Err(e) => println!("{}: {e:?}", path.display()),
                            }
                        }
                        Err(e) => println!("{}: {e:?}", path.display()),
                    }
                }
                Err(e) => println!("{:?}", e),
            }
        }
    }


    #[test]
    fn individual_test() {
        let mut text = include_str!("../../data/zy2.nl");

        match crate::token::parse::problem(&mut text) {
            Ok(intermediate) => {
                let input = Stateful::<&[Token], _> {
                    input: &intermediate.body,
                    state: &intermediate.headers,
                };
                match problem().parse(input) {
                    Ok(_p) => println!("{_p:#?}"),
                    Err(e) => println!("{e:?}"),
                }
            }
            Err(e) => println!("{e:?}"),
        }
    }
}
