from deepeval.metrics import GEval, FaithfulnessMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCaseParams

LLM_MODEL = "gpt-4o"


correctness_metric = GEval(
    name="Correctness",
    model = LLM_MODEL,
    criteria="Determine whether the actual output is factually correct based on the expected output.",
    # NOTE: you can only provide either criteria or evaluation_steps, and not both
    evaluation_steps=[
        "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
        "You should also heavily penalize omission of detail",
        #"Vague language, or contradicting OPINIONS, are OK"
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
)

faithfulness_metric = FaithfulnessMetric(
    threshold=0.7,
    model=LLM_MODEL,
    include_reason=False
)

relevance_metric = ContextualRelevancyMetric(
    # evaluates whether the text chunk size and top-K of your retriever is able to retrieve information without much irrelevancies.
    threshold=1,
    model=LLM_MODEL,
    include_reason=True
)

# Define evaluation metrics
correctness_metric = GEval(
    name="Correctness",
    model=LLM_MODEL,
    evaluation_params=[
        LLMTestCaseParams.EXPECTED_OUTPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT
    ],
    evaluation_steps=[
        "Determine whether the actual output is factually correct based on the expected output."
    ],
)

