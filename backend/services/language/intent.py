from enum import Enum


class Intent(str, Enum):
    UNKNOWN = "unknown"

    # Core computation
    CALCULATE = "calculate"
    SIMPLIFY = "simplify"
    EXPAND = "expand"
    FACTOR = "factor"

    # Calculus
    DIFFERENTIATE = "differentiate"
    INTEGRATE = "integrate"
    LIMIT = "limit"

    # Equations
    SOLVE = "solve"
    ROOTS = "roots"
    MAXIMUM = "maximum"
    MINIMUM = "minimum"

    # Algebra / matrices
    MATRIX = "matrix"
    DETERMINANT = "determinant"
    INVERSE = "inverse"

    # Statistics
    MEAN = "mean"
    MEDIAN = "median"
    VARIANCE = "variance"
    STANDARD_DEVIATION = "standard_deviation"

    # Probability
    PROBABILITY = "probability"

    # Graphing
    PLOT = "plot"

    # Conversation / follow-ups
    FOLLOW_UP = "follow_up"
    EXPLAIN = "explain"
    ALTERNATIVE_METHOD = "alternative_method"

    # Learning
    GENERATE_MCQ = "generate_mcq"
    START_MCQ = "start_mcq"
    NEXT_MCQ = "next_mcq"
    HINT = "hint"
    SUBMIT_ANSWER = "submit_answer"
    SHOW_SOLUTION = "show_solution"

    # Practice
    SIMILAR_PROBLEM = "similar_problem"
    HARDER_PROBLEM = "harder_problem"
    EASIER_PROBLEM = "easier_problem"