PREFETCH_BUDGET_DICT_NQ_SMALL = {
    "h100": {
        "linear": 10.0,
        "parallel": 8.0,
        "iterative": 5.0,
        "iterretgen": 4.0,
        "flare": 6.0,
        "selfrag": 3,
    },
    "rtx4090": {
        "linear": 7.0,
        "parallel": 7.0,
        "iterative": 3,
        "iterretgen": 2.5,
        "flare": 3,
        "selfrag": 1.25,
    }
}

PREFETCH_BUDGET_DICT_HotpotQA_SMALL = {
    "h100": {
        "linear": 10.0,
        "parallel": 9.0,
        "iterative": 7.0,
        "iterretgen": 3.0,
        "flare": 7.0,
        "selfrag": 4.5,
    },
    "rtx4090": {
        "linear": 7.0,
        "parallel": 7.0,
        "iterative": 4.5,
        "iterretgen": 2.0,
        "flare": 2.5,
        "selfrag": 3,
    }
}

PREFETCH_BUDGET_DICT_TriviaQA_SMALL = {
    "h100": {
        "linear": 10.0,
        "parallel": 9.0,
        "iterative": 7.0,
        "iterretgen": 3.0,
        "flare": 6.0,
        "selfrag": 4.5,
    },
    "rtx4090": {
        "linear": 7.0,
        "parallel": 7.0,
        "iterative": 4.25,
        "iterretgen": 2.0,
        "flare": 2.0,
        "selfrag": 2.75,
    }
}

PREFETCH_BUDGET_DICT_SMALL = {
    "nq": PREFETCH_BUDGET_DICT_NQ_SMALL,
    "hotpotqa": PREFETCH_BUDGET_DICT_HotpotQA_SMALL,
    "triviaqa": PREFETCH_BUDGET_DICT_TriviaQA_SMALL,
}

PREFETCH_BUDGET_DICT_NQ_LARGE = {
    "h100": {
        "linear": 10.0,
        "parallel": 9.0,
        "iterative": 5.0,
        "iterretgen": 4.0,
        "flare": 6.0,
        "selfrag": 3,
    },
    "rtx4090": {
        "linear": 7.0,
        "parallel": 7.0,
        "iterative": 3,
        "iterretgen": 2.5,
        "flare": 3,
        "selfrag": 1.25,
    }
}

PREFETCH_BUDGET_DICT_HotpotQA_LARGE = {
    "h100": {
        "linear": 16.0,
        "parallel": 16.0,
        "iterative": 8.0,
        "iterretgen": 8.0,
        "flare": 9.0,
        "selfrag": 7,
    },
    "rtx4090": {
        "linear": 7.0,
        "parallel": 7.0,
        "iterative": 4.5,
        "iterretgen": 2.0,
        "flare": 2.5,
        "selfrag": 3,
    }
}

PREFETCH_BUDGET_DICT_TriviaQA_LARGE = {
    "h100": {
        "linear": 16.0,
        "parallel": 16.0,
        "iterative": 8.0,
        "iterretgen": 8.0,
        "flare": 6.5,
        "selfrag": 7,
    },
    "rtx4090": {
        "linear": 7.0,
        "parallel": 7.0,
        "iterative": 4.25,
        "iterretgen": 2.0,
        "flare": 2.0,
        "selfrag": 2.75,
    }
}

PREFETCH_BUDGET_DICT_NQ_LARGE = {
    "h100": {
        "linear": 10.0,
        "parallel": 8.0,
        "iterative": 5.0,
        "iterretgen": 4.0,
        "flare": 6.0,
        "selfrag": 3,
    },
    "rtx4090": {
        "linear": 7.0,
        "parallel": 7.0,
        "iterative": 3,
        "iterretgen": 2.5,
        "flare": 3,
        "selfrag": 1.25,
    }
}

PREFETCH_BUDGET_DICT_HotpotQA_LARGE = {
    "h100": {
        "linear": 10.0,
        "parallel": 9.0,
        "iterative": 7.0,
        "iterretgen": 3.0,
        "flare": 7.0,
        "selfrag": 4.5,
    },
    "rtx4090": {
        "linear": 7.0,
        "parallel": 7.0,
        "iterative": 4.5,
        "iterretgen": 2.0,
        "flare": 2.5,
        "selfrag": 3,
    }
}

PREFETCH_BUDGET_DICT_TriviaQA_LARGE = {
    "h100": {
        "linear": 10.0,
        "parallel": 9.0,
        "iterative": 7.0,
        "iterretgen": 3.0,
        "flare": 6.0,
        "selfrag": 4.5,
    },
    "rtx4090": {
        "linear": 7.0,
        "parallel": 7.0,
        "iterative": 4.25,
        "iterretgen": 2.0,
        "flare": 2.0,
        "selfrag": 2.75,
    }
}

PREFETCH_BUDGET_DICT_LARGE = {
    "nq": PREFETCH_BUDGET_DICT_NQ_LARGE,
    "hotpotqa": PREFETCH_BUDGET_DICT_HotpotQA_LARGE,
    "triviaqa": PREFETCH_BUDGET_DICT_TriviaQA_LARGE,
}

PREFETCH_BUDGET_DICT_NQ_22B = {
    "h100": {
        "linear": 18.0,
        "parallel": 18.0,
        "iterative": 7.0,
        "iterretgen": 12.0,
        "flare": 12.0,
        "selfrag": 4.5,
    },
}

PREFETCH_BUDGET_DICT_HotpotQA_22B = {
    "h100": {
        "linear": 18.0,
        "parallel": 18.0,
        "iterative": 12.0,
        "iterretgen": 10.0,
        "flare": 10.0,
        "selfrag": 9.0,
    },
}

PREFETCH_BUDGET_DICT_TriviaQA_22B = {
    "h100": {
        "linear": 18.0,
        "parallel": 18.0,
        "iterative": 10.0,
        "iterretgen": 9.0,
        "flare": 8.0,
        "selfrag": 8.0,
    },
}

PREFETCH_BUDGET_DICT_22B = {
    "nq": PREFETCH_BUDGET_DICT_NQ_22B,
    "hotpotqa": PREFETCH_BUDGET_DICT_HotpotQA_22B,
    "triviaqa": PREFETCH_BUDGET_DICT_TriviaQA_22B,
}
# PREFETCH_BUDGET_DICT = {
#     "h100": {
#         "linear": 8.0,
#         "iterative": 5.0,
#         "iterretgen": 4.0,
#         "parallel": 8.0,
#         "flare": 6.0,
#         "selfrag": 3,
#     },
#     "rtx4090": {
#         "linear": 6.0,
#         "iterative": 2.5,
#         "iterretgen": 1.75,
#         "parallel": 6.0,
#         "flare": 3.5,
#         "selfrag": 1.25,
#     }
# }
