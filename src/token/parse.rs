use winnow::{
    ascii::{
        alpha0 as alphas, dec_int, dec_uint, float, line_ending, not_line_ending, space0 as spaces,
        space1 as any_space,
    },
    binary::length_take,
    combinator::{alt, eof, fail, opt, peek, preceded, repeat, separated, success, terminated},
    dispatch,
    error::InputError,
    seq,
    token::any,
    trace::trace,
    PResult, Parser,
};

use super::*;

#[allow(unused)]
pub fn problem<'a>(input: &mut &'a str) -> PResult<Problem, InputError<&'a str>> {
    seq! {Problem {
        headers: headers,
        _: line_ending,
        body: repeat(0.., terminated(token, line_ending))
    }}
    .parse_next(input)
}

fn headers<'a>(input: &mut &'a str) -> PResult<Headers, InputError<&'a str>> {
    seq! {Headers {
        problem: problem_statistics,
        _: (opt((spaces, comment)), line_ending, spaces),
        general: general_statistics,
        _: (opt((spaces, comment)), line_ending, spaces),
        nonlinear: nonlinear_statistics,
        _: (opt((spaces, comment)), line_ending, spaces),
        network_constraints: network_constraint_statistics,
        _: (opt((spaces, comment)), line_ending, spaces),
        nonlinear_vars: nonlinear_variable_statistics,
        _: (opt((spaces, comment)), line_ending, spaces),
        linear_network: linear_network_statistics,
        _: (opt((spaces, comment)), line_ending, spaces),
        discrete_variables: discrete_variable_statistics,
        _: (opt((spaces, comment)), line_ending, spaces),
        nonzeros: nonzero_statistics,
        _: (opt((spaces, comment)), line_ending, spaces),
        name_lengths: max_name_length_statistics,
        _: (opt((spaces, comment)), line_ending, spaces),
        common_exprs: common_expression_statistics,
        _: opt((spaces, comment)),
    }}
    .parse_next(input)
}

fn comment<'a>(input: &mut &'a str) -> PResult<Comment, InputError<&'a str>> {
    seq! {Comment{
        _: ('#', spaces),
        body: not_line_ending.map(str::to_owned)
    }}
    .parse_next(input)
}

fn header_numbers<'a>(input: &mut &'a str) -> PResult<Vec<u64>, InputError<&'a str>> {
    trace(
        "header_numbers",
        separated(0.., float::<_, u64, _>, any_space),
    )
    .parse_next(input)
}

fn problem_statistics<'a>(input: &mut &'a str) -> PResult<ProblemStatistics, InputError<&'a str>> {
    seq! {ProblemStatistics {
        _: 'g',
        nums: header_numbers
    }}
    .parse_next(input)
}

fn general_statistics<'a>(input: &mut &'a str) -> PResult<GeneralStatistics, InputError<&'a str>> {
    seq! { GeneralStatistics {
        vars: dec_uint,
        _: spaces,
        constraints: dec_uint,
        _: spaces,
        objectives: dec_uint,
        _: spaces,
        ranges: dec_uint,
        _: spaces,
        lcons: dec_uint,
    }}
    .parse_next(input)
}
fn nonlinear_statistics<'a>(
    input: &mut &'a str,
) -> PResult<NonlinearStatistics, InputError<&'a str>> {
    seq! { NonlinearStatistics {
        constraints: dec_uint,
        _: spaces,
        objectives: dec_uint,
    }}
    .parse_next(input)
}
fn network_constraint_statistics<'a>(
    input: &mut &'a str,
) -> PResult<NetworkConstraintStatistics, InputError<&'a str>> {
    seq! { NetworkConstraintStatistics {
        nonlinear: dec_uint,
        _: spaces,
        linear: dec_uint
    }}
    .parse_next(input)
}
fn nonlinear_variable_statistics<'a>(
    input: &mut &'a str,
) -> PResult<NonlinearVariableStatistics, InputError<&'a str>> {
    seq! { NonlinearVariableStatistics {
        constraints: dec_uint,
        _: spaces,
        objectives: dec_uint,
        _: spaces,
        both: dec_uint,
    }}
    .parse_next(input)
}
fn linear_network_statistics<'a>(
    input: &mut &'a str,
) -> PResult<LinearNetworkStatistics, InputError<&'a str>> {
    seq! { LinearNetworkStatistics {
        variables: dec_uint,
        _: spaces,
        functions: dec_uint,
        _: spaces,
        arith: dec_uint,
        _: spaces,
        flags: dec_uint,
    }}
    .parse_next(input)
}
fn discrete_variable_statistics<'a>(
    input: &mut &'a str,
) -> PResult<DiscreteVariableStatistics, InputError<&'a str>> {
    seq! { DiscreteVariableStatistics {
        binary: dec_uint,
        _: spaces,
        integer: dec_uint,
        _: spaces,
        nonlinear_b: dec_uint,
        _: spaces,
        nonlinear_c: dec_uint,
        _: spaces,
        nonlinear_o: dec_uint,
    }}
    .parse_next(input)
}
fn nonzero_statistics<'a>(input: &mut &'a str) -> PResult<NonzeroStatistics, InputError<&'a str>> {
    seq! { NonzeroStatistics {
        jacobian: dec_uint,
        _: spaces,
        gradients: dec_uint,
    }}
    .parse_next(input)
}
fn max_name_length_statistics<'a>(
    input: &mut &'a str,
) -> PResult<MaxNameLengthStatistics, InputError<&'a str>> {
    seq! { MaxNameLengthStatistics {
        constraints: dec_uint,
        _: spaces,
        variables: dec_uint,
    }}
    .parse_next(input)
}
fn common_expression_statistics<'a>(
    input: &mut &'a str,
) -> PResult<CommonExpressionStatistics, InputError<&'a str>> {
    seq! { CommonExpressionStatistics {
        b: dec_uint,
        _: spaces,
        c: dec_uint,
        _: spaces,
        o: dec_uint,
        _: spaces,
        c1: dec_uint,
        _: spaces,
        o1: dec_uint,
    }}
    .parse_next(input)
}

fn token<'a>(input: &mut &'a str) -> PResult<Token, InputError<&'a str>> {
    trace(
        "Token",
        dispatch! {peek(any);
                'n' => trace("number", number),
                'v' => trace("variable", variable),
                'f' => trace("function", function),
                'h' => trace("string", string),
                'o' => trace("operation", operation),
                '0'..='9' => trace("numbers", numbers),
                seg => trace("segment", segment(seg)),
        },
    )
    .parse_next(input)
}

fn number<'a>(input: &mut &'a str) -> PResult<Token, InputError<&'a str>> {
    trace("number", preceded('n', float).map(Token::Number)).parse_next(input)
}

fn numbers<'a>(input: &mut &'a str) -> PResult<Token, InputError<&'a str>> {
    separated(1..=3, numbers_number, any_space)
        .map(|collected: Vec<_>| match collected.len() {
            1 => Token::Single(collected[0]),
            2 => Token::Pair(collected[0], collected[1]),
            3 => Token::Trio(collected[0], collected[1], collected[2]),
            _ => unreachable!(),
        })
        .parse_next(input)
}

fn numbers_number<'a>(input: &mut &'a str) -> PResult<Number, InputError<&'a str>> {
    alt((
        terminated(dec_int, peek(alt((any_space, eof, line_ending)))).map(Number::Integer),
        float.map(Number::Real),
    ))
    .parse_next(input)
}

fn variable<'a>(input: &mut &'a str) -> PResult<Token, InputError<&'a str>> {
    trace("variable", preceded('v', dec_uint).map(Token::Variable)).parse_next(input)
}

fn function<'a>(input: &mut &'a str) -> PResult<Token, InputError<&'a str>> {
    trace(
        "imported function",
        preceded('f', (dec_uint, spaces, dec_uint))
            .map(|(num1, _, num2)| Token::Function(num1, num2)),
    )
    .parse_next(input)
}

fn string<'a>(input: &mut &'a str) -> PResult<Token, InputError<&'a str>> {
    trace(
        "string",
        preceded(
            'h',
            length_take(terminated(dec_uint::<_, u64, _>, ':'))
                .map(|out: &'a str| Token::String(out.to_owned())),
        ),
    )
    .parse_next(input)
}

fn operation<'a>(input: &mut &'a str) -> PResult<Token, InputError<&'a str>> {
    trace("operation", preceded('o', dispatch! {dec_uint;
        0 => trace("Operation::ADD", success(Token::Operation(Operator::Add))),
        1 => trace("Operation::SUB", success(Token::Operation(Operator::Sub))),
        2 => trace("Operation::MUL", success(Token::Operation(Operator::Mul))),
        3 => trace("Operation::DIV", success(Token::Operation(Operator::Div))),
        4 => trace("Operation::MOD", success(Token::Operation(Operator::Mod))),
        5 => trace("Operation::POW", success(Token::Operation(Operator::Pow))),
        6 => trace("Operation::LESS", success(Token::Operation(Operator::Less))),
        11 => trace("Operation::MIN", success(Token::Operation(Operator::Min))),
        12 => trace("Operation::MAX", success(Token::Operation(Operator::Max))),
        13 => trace("Operation::FLOOR", success(Token::Operation(Operator::Floor))),
        14 => trace("Operation::CEIL", success(Token::Operation(Operator::Ceiling))),
        15 => trace("Operation::ABS", success(Token::Operation(Operator::AbsoluteValue))),
        16 => trace("Operation::MINUS", success(Token::Operation(Operator::UnaryMinus))),
        20 => trace("Operation::OR", success(Token::Operation(Operator::Or))),
        21 => trace("Operation::AND", success(Token::Operation(Operator::And))),
        22 => trace("Operation::LT", success(Token::Operation(Operator::LessThan))),
        23 => trace("Operation::LE", success(Token::Operation(Operator::LessThanOrEquals))),
        24 => trace("Operation::EQ", success(Token::Operation(Operator::Equals))),
        28 => trace("Operation::GE", success(Token::Operation(Operator::GreatherThanOrEquals))),
        29 => trace("Operation::GT", success(Token::Operation(Operator::GreatherThan))),
        30 => trace("Operation::NE", success(Token::Operation(Operator::NotEquals))),
        34 => trace("Operation::NOT", success(Token::Operation(Operator::Not))),
        35 => trace("Operation::IF", success(Token::Operation(Operator::If))),
        37 => trace("Operation::TANH", success(Token::Operation(Operator::HyperbolicTangent))),
        38 => trace("Operation::TAN", success(Token::Operation(Operator::Tangent))),
        39 => trace("Operation::SQRT", success(Token::Operation(Operator::SquareRoot))),
        40 => trace("Operation::SINH", success(Token::Operation(Operator::HyperbolicSine))),
        41 => trace("Operation::SIN", success(Token::Operation(Operator::Sine))),
        42 => trace("Operation::LOG10", success(Token::Operation(Operator::LogarithmBase10))),
        43 => trace("Operation::LOG", success(Token::Operation(Operator::NaturalLogarithm))),
        44 => trace("Operation::EXP", success(Token::Operation(Operator::Exponential))),
        45 => trace("Operation::COSH", success(Token::Operation(Operator::HyperbolicCosine))),
        46 => trace("Operation::COS", success(Token::Operation(Operator::Cosine))),
        47 => trace("Operation::ATANH", success(Token::Operation(Operator::InverseHyperbolicTangent))),
        48 => trace("Operation::ATAN2", success(Token::Operation(Operator::InverseTangent2))),
        49 => trace("Operation::ATAN", success(Token::Operation(Operator::InverseTangent))),
        50 => trace("Operation::ASINH", success(Token::Operation(Operator::InverseHyperbolicSine))),
        51 => trace("Operation::ASIN", success(Token::Operation(Operator::InverseSine))),
        52 => trace("Operation::ACOSH", success(Token::Operation(Operator::InverseHyperbolicCosine))),
        53 => trace("Operation::ACOS", success(Token::Operation(Operator::InverseCosine))),
        54 => trace("Operation::SUM", success(Token::Operation(Operator::Sum))),
        55 => trace("Operation::TRUNC_DIV", success(Token::Operation(Operator::TruncatedDivision))),
        56 => trace("Operation::PRECISION", success(Token::Operation(Operator::Precision))),
        57 => trace("Operation::ROUND", success(Token::Operation(Operator::Round))),
        58 => trace("Operation::TRUNC", success(Token::Operation(Operator::Truncate))),
        59 => trace("Operation::COUNT", success(Token::Operation(Operator::Count))),
        60 => trace("Operation::NUMBEROF", success(Token::Operation(Operator::NumberOf))),
        61 => trace("Operation::NUMBEROF_SYM", success(Token::Operation(Operator::SymbolicNumberOf))),
        62 => trace("Operation::ATLEAST", success(Token::Operation(Operator::AtLeast))),
        63 => trace("Operation::ATMOST", success(Token::Operation(Operator::AtMost))),
        64 => trace("Operation::PLTERM", success(Token::Operation(Operator::PiecewiseLinearTerm))),
        65 => trace("Operation::IFSYM", success(Token::Operation(Operator::SymbolicIf))),
        66 => trace("Operation::EXACTLY", success(Token::Operation(Operator::Exactly))),
        67 => trace("Operation::NOT_ATLEAST", success(Token::Operation(Operator::NotAtLeast))),
        68 => trace("Operation::NOT_ATMOST", success(Token::Operation(Operator::NotAtMost))),
        69 => trace("Operation::NOT_EXACTLY", success(Token::Operation(Operator::NotExactly))),
        70 => trace("Operation::FORALL", success(Token::Operation(Operator::ForAll))),
        71 => trace("Operation::EXISTS", success(Token::Operation(Operator::Exists))),
        72 => trace("Operation::IMPLICATION", success(Token::Operation(Operator::Implies))),
        73 => trace("Operation::IFF", success(Token::Operation(Operator::IfAndOnlyIf))),
        74 => trace("Operation::ALLDIFF", success(Token::Operation(Operator::AllDifferent))),
        75 => trace("Operation::NOT_ALLDIFF", success(Token::Operation(Operator::NotAllDifferent))),
        76 => trace("Operation::POW_CONST_EXP", success(Token::Operation(Operator::PowerConstantExponent))),
        77 => trace("Operation::POW2", success(Token::Operation(Operator::Square))),
        78 => trace("Operation::POW_CONST_BASE", success(Token::Operation(Operator::PowerConstantBase))),
        _ => fail
    })).parse_next(input)
}

fn segment<'a>(typ: char) -> impl FnMut(&mut &'a str) -> PResult<Token, InputError<&'a str>> {
    match typ {
        'F' => |input: &mut &'a str| imported_function_segment(input),
        'S' => |input: &mut &'a str| suffix_segment(input),
        'V' => |input: &mut &'a str| variable_definition_segment(input),
        'C' => |input: &mut &'a str| algebraic_constraint_segment(input),
        'L' => |input: &mut &'a str| logical_constraint_segment(input),
        'O' => |input: &mut &'a str| objective_function_segment(input),
        'd' => |input: &mut &'a str| dual_guess_segment(input),
        'x' => |input: &mut &'a str| primal_guess_segment(input),
        'r' => |input: &mut &'a str| algebraic_constraint_bounds_segment(input),
        'b' => |input: &mut &'a str| variable_bounds_segment(input),
        'k' => |input: &mut &'a str| jacobian_column_counts_segment(input),
        'J' => |input: &mut &'a str| jacobian_sparsity_segment(input),
        'G' => |input: &mut &'a str| gradient_sparsity_segment(input),
        _ => fail,
    }
}

fn imported_function_segment<'a>(input: &mut &'a str) -> PResult<Token, InputError<&'a str>> {
    preceded(
        'F',
        (dec_uint, spaces, dec_uint, spaces, dec_uint, spaces, alphas).map(
            |(id, _, j, _, argument_count, _, name): (_, _, u8, _, _, _, &'a str)| {
                Token::Segment(SegmentId::ImportedFunction(ImportedFunctionMetadata {
                    id,
                    string_arguments_allowed: j == 1,
                    argument_count,
                    name: name.to_owned(),
                }))
            },
        ),
    )
    .parse_next(input)
}

fn suffix_segment<'a>(input: &mut &'a str) -> PResult<Token, InputError<&'a str>> {
    preceded(
        'S',
        (dec_uint, spaces, dec_uint, spaces, alphas).map(
            |(bitfield, _, count, _, name): (u8, _, _, _, &'a str)| {
                Token::Segment(SegmentId::Suffix(SuffixMetadata {
                    kind: SuffixKind::from(bitfield & 3),
                    real: bitfield & 4 == 4,
                    count,
                    name: name.to_owned(),
                }))
            },
        ),
    )
    .parse_next(input)
}

fn variable_definition_segment<'a>(input: &mut &'a str) -> PResult<Token, InputError<&'a str>> {
    preceded(
        'V',
        (dec_uint, spaces, dec_uint, spaces, dec_uint).map(
            |(id, _, linear_terms, _, constraint_info)| {
                Token::Segment(SegmentId::VariableDefinition(VariableDefinitionMetadata {
                    id,
                    linear_terms,
                    constraint_info,
                }))
            },
        ),
    )
    .parse_next(input)
}

fn algebraic_constraint_segment<'a>(input: &mut &'a str) -> PResult<Token, InputError<&'a str>> {
    preceded(
        'C',
        dec_uint.map(|id| {
            Token::Segment(SegmentId::AlgebraicConstraint(
                AlgebraicConstraintMetadata { id },
            ))
        }),
    )
    .parse_next(input)
}

fn logical_constraint_segment<'a>(input: &mut &'a str) -> PResult<Token, InputError<&'a str>> {
    preceded(
        'L',
        dec_uint.map(|id| {
            Token::Segment(SegmentId::LogicalConstraint(LogicalConstraintMetadata {
                id,
            }))
        }),
    )
    .parse_next(input)
}

fn objective_function_segment<'a>(input: &mut &'a str) -> PResult<Token, InputError<&'a str>> {
    preceded(
        'O',
        (dec_uint, spaces, dec_uint).map(|(id, _, typ)| {
            Token::Segment(SegmentId::ObjectiveFunction(ObjectiveFunctionMetadata {
                id,
                typ,
            }))
        }),
    )
    .parse_next(input)
}

fn dual_guess_segment<'a>(input: &mut &'a str) -> PResult<Token, InputError<&'a str>> {
    preceded(
        'd',
        dec_uint.map(|count| {
            Token::Segment(SegmentId::DualInitialGuess(DualInitialGuessMetadata {
                count,
            }))
        }),
    )
    .parse_next(input)
}

fn primal_guess_segment<'a>(input: &mut &'a str) -> PResult<Token, InputError<&'a str>> {
    preceded(
        'x',
        dec_uint.map(|count| {
            Token::Segment(SegmentId::PrimalInitialGuess(PrimalInitialGuessMetadata {
                count,
            }))
        }),
    )
    .parse_next(input)
}

fn algebraic_constraint_bounds_segment<'a>(
    input: &mut &'a str,
) -> PResult<Token, InputError<&'a str>> {
    'r'.map(|_| Token::Segment(SegmentId::AlgebraicConstraintBounds))
        .parse_next(input)
}

fn variable_bounds_segment<'a>(input: &mut &'a str) -> PResult<Token, InputError<&'a str>> {
    'b'.map(|_| Token::Segment(SegmentId::VariableBounds))
        .parse_next(input)
}

fn jacobian_column_counts_segment<'a>(input: &mut &'a str) -> PResult<Token, InputError<&'a str>> {
    preceded(
        'k',
        dec_uint.map(|count| {
            Token::Segment(SegmentId::JacobianColumnCounts(
                JacobianColumnCountsMetadata { count },
            ))
        }),
    )
    .parse_next(input)
}

fn jacobian_sparsity_segment<'a>(input: &mut &'a str) -> PResult<Token, InputError<&'a str>> {
    preceded(
        'J',
        (dec_uint, spaces, dec_uint).map(|(id, _, count)| {
            Token::Segment(SegmentId::Jacobian(JacobianMetadata { id, count }))
        }),
    )
    .parse_next(input)
}

fn gradient_sparsity_segment<'a>(input: &mut &'a str) -> PResult<Token, InputError<&'a str>> {
    preceded(
        'G',
        (dec_uint, spaces, dec_uint).map(|(id, _, count)| {
            Token::Segment(SegmentId::Gradient(GradientMetadata { id, count }))
        }),
    )
    .parse_next(input)
}

#[cfg(test)]
mod tests {

    use std::{fs::File, io::Read};

    use super::*;

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

                    println!("{path:?}");

                    match problem(&mut content.as_str()) {
                        Ok(_) => {}
                        Err(e) => println!("{:?}: {e:?}", path.display()),
                    }
                }
                Err(e) => println!("{:?}", e),
            }
        }
    }

    #[test]
    fn individual_test() {
        let mut text = include_str!("../../data/yfit.nl");

        match problem(&mut text) {
            Ok(_) => {}
            Err(e) => println!("{e:?}"),
        }
    }

    #[test]
    fn test_comment() {
        let mut text = "# problem 3pk";
        let expect = Comment {
            body: "problem 3pk".to_owned(),
        };
        assert_eq!(comment(&mut text), Ok(expect));
    }

    #[test]
    fn test_header_numbers() {
        let mut text = "0 1 2 3 4 5";
        let expect: Vec<u64> = vec![0, 1, 2, 3, 4, 5];

        assert_eq!(header_numbers(&mut text), Ok(expect));
    }

    #[test]
    fn test_problem_statistics() {
        let mut text = "g3 0 1 0";
        let expect = ProblemStatistics {
            nums: vec![3, 0, 1, 0],
        };

        assert_eq!(problem_statistics(&mut text), Ok(expect));
    }

    #[test]
    fn test_general_statistics() {
        let mut text = "1 2 3 4 5";
        let expect = GeneralStatistics {
            vars: 1,
            constraints: 2,
            objectives: 3,
            ranges: 4,
            lcons: 5,
        };

        assert_eq!(general_statistics(&mut text), Ok(expect));
    }

    #[test]
    fn test_nonlinear_statistics() {
        let mut text = "2 3";
        let expect = NonlinearStatistics {
            constraints: 2,
            objectives: 3,
        };

        assert_eq!(nonlinear_statistics(&mut text), Ok(expect));
    }

    #[test]
    fn test_network_constraint_statistics() {
        let mut text = "2 3";
        let expect = NetworkConstraintStatistics {
            nonlinear: 2,
            linear: 3,
        };

        assert_eq!(network_constraint_statistics(&mut text), Ok(expect));
    }

    #[test]
    fn test_nonlinear_variable_statistics() {
        let mut text = "2 3 4";
        let expect = NonlinearVariableStatistics {
            constraints: 2,
            objectives: 3,
            both: 4,
        };

        assert_eq!(nonlinear_variable_statistics(&mut text), Ok(expect));
    }

    #[test]
    fn test_linear_network_statistics() {
        let mut text = "1 2 3 4";
        let expect = LinearNetworkStatistics {
            variables: 1,
            functions: 2,
            arith: 3,
            flags: 4,
        };

        assert_eq!(linear_network_statistics(&mut text), Ok(expect));
    }

    #[test]
    fn test_discrete_variable_statistics() {
        let mut text = "0 1 2 3 4";
        let expect = DiscreteVariableStatistics {
            binary: 0,
            integer: 1,
            nonlinear_b: 2,
            nonlinear_c: 3,
            nonlinear_o: 4,
        };

        assert_eq!(discrete_variable_statistics(&mut text), Ok(expect));
    }

    #[test]
    fn test_nonzero_statistics() {
        let mut text = "0 1";
        let expect = NonzeroStatistics {
            jacobian: 0,
            gradients: 1,
        };

        assert_eq!(nonzero_statistics(&mut text), Ok(expect));
    }

    #[test]
    fn test_max_name_length_statistics() {
        let mut text = "0 1";
        let expect = MaxNameLengthStatistics {
            constraints: 0,
            variables: 1,
        };

        assert_eq!(max_name_length_statistics(&mut text), Ok(expect));
    }

    #[test]
    fn test_common_expression_statistics() {
        let mut text = "1 2 3 4 5";
        let expect = CommonExpressionStatistics {
            b: 1,
            c: 2,
            o: 3,
            c1: 4,
            o1: 5,
        };

        assert_eq!(common_expression_statistics(&mut text), Ok(expect));
    }

    #[test]
    fn test_token() {
        let mut cases = vec![
            ("n0", Token::Number(0.0)),
            ("v0", Token::Variable(0)),
            ("f0 1", Token::Function(0, 1)),
            ("h4:test", Token::String("test".to_owned())),
            ("o0", Token::Operation(Operator::Add)),
            ("1 2", Token::Pair(Number::Integer(1), Number::Integer(2))),
            (
                "x30",
                Token::Segment(SegmentId::PrimalInitialGuess(PrimalInitialGuessMetadata {
                    count: 30,
                })),
            ),
        ];

        for (case, expect) in cases.iter_mut() {
            assert_eq!(token(case), Ok(expect.clone()));
        }
    }

    #[test]
    fn test_number() {
        let mut text = "n-1.5e-6";
        let expect = Token::Number(-1.5e-6);

        assert_eq!(number(&mut text), Ok(expect));
    }

    #[test]
    fn test_numbers() {
        let mut cases = vec![
            ("0", Token::Single(Number::Integer(0))),
            ("0 1", Token::Pair(Number::Integer(0), Number::Integer(1))),
            (
                "0 1 2",
                Token::Trio(Number::Integer(0), Number::Integer(1), Number::Integer(2)),
            ),
        ];

        for (case, expect) in cases.iter_mut() {
            assert_eq!(numbers(case), Ok(expect.clone()));
        }
    }

    #[test]
    fn test_numbers_number() {
        let mut cases = [("0", Number::Integer(0)), ("0.0", Number::Real(0.0))];

        for (case, expect) in cases.iter_mut() {
            assert_eq!(numbers_number(case), Ok(*expect));
        }
    }

    #[test]
    fn test_variable() {
        let mut text = "v15";
        let expect = Token::Variable(15);

        assert_eq!(variable(&mut text), Ok(expect));
    }

    #[test]
    fn test_imported_function() {
        let mut text = "f1 2";
        let expect = Token::Function(1, 2);

        assert_eq!(function(&mut text), Ok(expect));
    }

    #[test]
    fn test_string() {
        let mut text = "h4:test";
        let expect = Token::String("test".to_owned());

        assert_eq!(string(&mut text), Ok(expect));
    }

    #[test]
    fn test_operation() {
        let mut cases = vec![
            ("o0", Token::Operation(Operator::Add)),
            ("o1", Token::Operation(Operator::Sub)),
            ("o2", Token::Operation(Operator::Mul)),
            ("o3", Token::Operation(Operator::Div)),
            ("o4", Token::Operation(Operator::Mod)),
            ("o5", Token::Operation(Operator::Pow)),
            ("o6", Token::Operation(Operator::Less)),
            ("o11", Token::Operation(Operator::Min)),
            ("o12", Token::Operation(Operator::Max)),
            ("o13", Token::Operation(Operator::Floor)),
            ("o14", Token::Operation(Operator::Ceiling)),
            ("o15", Token::Operation(Operator::AbsoluteValue)),
            ("o16", Token::Operation(Operator::UnaryMinus)),
            ("o20", Token::Operation(Operator::Or)),
            ("o21", Token::Operation(Operator::And)),
            ("o22", Token::Operation(Operator::LessThan)),
            ("o23", Token::Operation(Operator::LessThanOrEquals)),
            ("o24", Token::Operation(Operator::Equals)),
            ("o28", Token::Operation(Operator::GreatherThanOrEquals)),
            ("o29", Token::Operation(Operator::GreatherThan)),
            ("o30", Token::Operation(Operator::NotEquals)),
            ("o34", Token::Operation(Operator::Not)),
            ("o35", Token::Operation(Operator::If)),
            ("o37", Token::Operation(Operator::HyperbolicTangent)),
            ("o38", Token::Operation(Operator::Tangent)),
            ("o39", Token::Operation(Operator::SquareRoot)),
            ("o40", Token::Operation(Operator::HyperbolicSine)),
            ("o41", Token::Operation(Operator::Sine)),
            ("o42", Token::Operation(Operator::LogarithmBase10)),
            ("o43", Token::Operation(Operator::NaturalLogarithm)),
            ("o44", Token::Operation(Operator::Exponential)),
            ("o45", Token::Operation(Operator::HyperbolicCosine)),
            ("o46", Token::Operation(Operator::Cosine)),
            ("o47", Token::Operation(Operator::InverseHyperbolicTangent)),
            ("o48", Token::Operation(Operator::InverseTangent2)),
            ("o49", Token::Operation(Operator::InverseTangent)),
            ("o50", Token::Operation(Operator::InverseHyperbolicSine)),
            ("o51", Token::Operation(Operator::InverseSine)),
            ("o52", Token::Operation(Operator::InverseHyperbolicCosine)),
            ("o53", Token::Operation(Operator::InverseCosine)),
            ("o54", Token::Operation(Operator::Sum)),
            ("o55", Token::Operation(Operator::TruncatedDivision)),
            ("o56", Token::Operation(Operator::Precision)),
            ("o57", Token::Operation(Operator::Round)),
            ("o58", Token::Operation(Operator::Truncate)),
            ("o59", Token::Operation(Operator::Count)),
            ("o60", Token::Operation(Operator::NumberOf)),
            ("o61", Token::Operation(Operator::SymbolicNumberOf)),
            ("o62", Token::Operation(Operator::AtLeast)),
            ("o63", Token::Operation(Operator::AtMost)),
            ("o64", Token::Operation(Operator::PiecewiseLinearTerm)),
            ("o65", Token::Operation(Operator::SymbolicIf)),
            ("o66", Token::Operation(Operator::Exactly)),
            ("o67", Token::Operation(Operator::NotAtLeast)),
            ("o68", Token::Operation(Operator::NotAtMost)),
            ("o69", Token::Operation(Operator::NotExactly)),
            ("o70", Token::Operation(Operator::ForAll)),
            ("o71", Token::Operation(Operator::Exists)),
            ("o72", Token::Operation(Operator::Implies)),
            ("o73", Token::Operation(Operator::IfAndOnlyIf)),
            ("o74", Token::Operation(Operator::AllDifferent)),
            ("o75", Token::Operation(Operator::NotAllDifferent)),
            ("o76", Token::Operation(Operator::PowerConstantExponent)),
            ("o77", Token::Operation(Operator::Square)),
            ("o78", Token::Operation(Operator::PowerConstantBase)),
        ];

        for (case, expect) in cases.iter_mut() {
            assert_eq!(token(case), Ok(expect.clone()));
        }
    }

    #[test]
    fn test_segment() {
        let mut cases = vec![
            (
                "F0 1 2 test",
                Token::Segment(SegmentId::ImportedFunction(ImportedFunctionMetadata {
                    id: 0,
                    string_arguments_allowed: true,
                    argument_count: 2,
                    name: "test".to_owned(),
                })),
            ),
            (
                "S0 1 test",
                Token::Segment(SegmentId::Suffix(SuffixMetadata {
                    kind: SuffixKind::Variables,
                    real: false,
                    count: 1,
                    name: "test".to_owned(),
                })),
            ),
            (
                "V0 1 2",
                Token::Segment(SegmentId::VariableDefinition(VariableDefinitionMetadata {
                    id: 0,
                    linear_terms: 1,
                    constraint_info: 2,
                })),
            ),
            (
                "C0",
                Token::Segment(SegmentId::AlgebraicConstraint(
                    AlgebraicConstraintMetadata { id: 0 },
                )),
            ),
            (
                "L0",
                Token::Segment(SegmentId::LogicalConstraint(LogicalConstraintMetadata {
                    id: 0,
                })),
            ),
            (
                "O0 1",
                Token::Segment(SegmentId::ObjectiveFunction(ObjectiveFunctionMetadata {
                    id: 0,
                    typ: 1,
                })),
            ),
            (
                "d0",
                Token::Segment(SegmentId::DualInitialGuess(DualInitialGuessMetadata {
                    count: 0,
                })),
            ),
            (
                "x0",
                Token::Segment(SegmentId::PrimalInitialGuess(PrimalInitialGuessMetadata {
                    count: 0,
                })),
            ),
            ("r", Token::Segment(SegmentId::AlgebraicConstraintBounds)),
            ("b", Token::Segment(SegmentId::VariableBounds)),
            (
                "k0",
                Token::Segment(SegmentId::JacobianColumnCounts(
                    JacobianColumnCountsMetadata { count: 0 },
                )),
            ),
            (
                "J0 1",
                Token::Segment(SegmentId::Jacobian(JacobianMetadata { id: 0, count: 1 })),
            ),
            (
                "G0 1",
                Token::Segment(SegmentId::Gradient(GradientMetadata { id: 0, count: 1 })),
            ),
        ];

        for (case, expect) in cases.iter_mut() {
            assert_eq!(token(case), Ok(expect.clone()));
        }
    }

    #[test]
    fn test_imported_function_segment() {
        let mut text = "F0 1 2 test";
        let expect = Token::Segment(SegmentId::ImportedFunction(ImportedFunctionMetadata {
            id: 0,
            string_arguments_allowed: true,
            argument_count: 2,
            name: "test".to_owned(),
        }));

        assert_eq!(imported_function_segment(&mut text), Ok(expect));
    }

    #[test]
    fn test_suffix_segment() {
        let mut text = "S0 1 test";
        let expect = Token::Segment(SegmentId::Suffix(SuffixMetadata {
            kind: SuffixKind::Variables,
            real: false,
            count: 1,
            name: "test".to_owned(),
        }));

        assert_eq!(suffix_segment(&mut text), Ok(expect));
    }

    #[test]
    fn test_variable_definition_segment() {
        let mut text = "V0 1 2";
        let expect = Token::Segment(SegmentId::VariableDefinition(VariableDefinitionMetadata {
            id: 0,
            linear_terms: 1,
            constraint_info: 2,
        }));

        assert_eq!(variable_definition_segment(&mut text), Ok(expect));
    }

    #[test]
    fn test_algebraic_constraint_segment() {
        let mut text = "C0";
        let expect = Token::Segment(SegmentId::AlgebraicConstraint(
            AlgebraicConstraintMetadata { id: 0 },
        ));

        assert_eq!(algebraic_constraint_segment(&mut text), Ok(expect));
    }

    #[test]
    fn test_logical_constraint_segment() {
        let mut text = "L0";
        let expect = Token::Segment(SegmentId::LogicalConstraint(LogicalConstraintMetadata {
            id: 0,
        }));

        assert_eq!(logical_constraint_segment(&mut text), Ok(expect));
    }

    #[test]
    fn test_objective_function_segment() {
        let mut text = "O0 1";
        let expect = Token::Segment(SegmentId::ObjectiveFunction(ObjectiveFunctionMetadata {
            id: 0,
            typ: 1,
        }));

        assert_eq!(objective_function_segment(&mut text), Ok(expect));
    }

    #[test]
    fn test_dual_guess_segment() {
        let mut text = "d0";
        let expect = Token::Segment(SegmentId::DualInitialGuess(DualInitialGuessMetadata {
            count: 0,
        }));

        assert_eq!(dual_guess_segment(&mut text), Ok(expect));
    }

    #[test]
    fn test_primal_guess_segment() {
        let mut text = "x0";
        let expect = Token::Segment(SegmentId::PrimalInitialGuess(PrimalInitialGuessMetadata {
            count: 0,
        }));

        assert_eq!(primal_guess_segment(&mut text), Ok(expect));
    }

    #[test]
    fn test_algebraic_constraint_bounds_segment() {
        let mut text = "r";
        let expect = Token::Segment(SegmentId::AlgebraicConstraintBounds);

        assert_eq!(algebraic_constraint_bounds_segment(&mut text), Ok(expect));
    }

    #[test]
    fn test_variable_bounds_segment() {
        let mut text = "b";
        let expect = Token::Segment(SegmentId::VariableBounds);

        assert_eq!(variable_bounds_segment(&mut text), Ok(expect));
    }

    #[test]
    fn test_jacobian_column_counts_segment() {
        let mut text = "k0";
        let expect = Token::Segment(SegmentId::JacobianColumnCounts(
            JacobianColumnCountsMetadata { count: 0 },
        ));

        assert_eq!(jacobian_column_counts_segment(&mut text), Ok(expect));
    }

    #[test]
    fn test_jacobian_sparsity_segment() {
        let mut text = "J0 1";
        let expect = Token::Segment(SegmentId::Jacobian(JacobianMetadata { id: 0, count: 1 }));

        assert_eq!(jacobian_sparsity_segment(&mut text), Ok(expect));
    }

    #[test]
    fn test_gradient_sparsity_segment() {
        let mut text = "G0 1";
        let expect = Token::Segment(SegmentId::Gradient(GradientMetadata { id: 0, count: 1 }));

        assert_eq!(gradient_sparsity_segment(&mut text), Ok(expect));
    }

    #[test]
    fn test_header() {
        let mut text = "g3 0 1 0	# problem 3pk
        30 0 1 0 0	# vars, constraints, objectives, ranges, eqns
        0 1	# nonlinear constraints, objectives
        0 0	# network constraints: nonlinear, linear
        0 30 0	# nonlinear vars in constraints, objectives, both
        0 0 0 1	# linear network variables; functions; arith, flags
        0 0 0 0 0	# discrete variables: binary, integer, nonlinear (b,c,o)
        0 30	# nonzeros in Jacobian, gradients
        0 0	# max name lengths: constraints, variables
        0 0 0 0 0	# common exprs: b,c,o,c1,o1";
        let expect = Headers {
            problem: ProblemStatistics {
                nums: vec![3, 0, 1, 0],
            },
            general: GeneralStatistics {
                vars: 30,
                constraints: 0,
                objectives: 1,
                ranges: 0,
                lcons: 0,
            },
            nonlinear: NonlinearStatistics {
                constraints: 0,
                objectives: 1,
            },
            network_constraints: NetworkConstraintStatistics {
                nonlinear: 0,
                linear: 0,
            },
            nonlinear_vars: NonlinearVariableStatistics {
                constraints: 0,
                objectives: 30,
                both: 0,
            },
            linear_network: LinearNetworkStatistics {
                variables: 0,
                functions: 0,
                arith: 0,
                flags: 1,
            },
            discrete_variables: DiscreteVariableStatistics {
                binary: 0,
                integer: 0,
                nonlinear_b: 0,
                nonlinear_c: 0,
                nonlinear_o: 0,
            },
            nonzeros: NonzeroStatistics {
                jacobian: 0,
                gradients: 30,
            },
            name_lengths: MaxNameLengthStatistics {
                constraints: 0,
                variables: 0,
            },
            common_exprs: CommonExpressionStatistics {
                b: 0,
                c: 0,
                o: 0,
                c1: 0,
                o1: 0,
            },
        };

        assert_eq!(headers(&mut text), Ok(expect));
    }
}
