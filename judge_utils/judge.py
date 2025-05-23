import asyncio
import warnings
from typing import Any, Dict, List, Tuple, Union

from fastchat.conversation import get_conv_template

from .prompts import (
    ABS_SYSTEM_PROMPT,
    ABSOLUTE_PROMPT_WO_REF,
    REL_SYSTEM_PROMPT,
    RELATIVE_PROMPT_WO_REF,
)
from .utils import async_batch_completions_with_retries, batch_completions_with_retries



class AGDEval:
    def __init__(
        self,
        model,
        relative_grade_template: str = RELATIVE_PROMPT_WO_REF,
    ):
        self.is_async = False  # Flag to indicate if the model is asynchronous

        if hasattr(model, "validate_vllm"):
            from .vllm import VLLM
        elif hasattr(model, "validate_litellm"):
            from .litellm import AsyncLiteLLM, LiteLLM

            if isinstance(model, AsyncLiteLLM):
                self.is_async = True
        elif hasattr(model, "validate_mockllm"):
            from .mock import AsyncMockLLM, MockLLM

            if isinstance(model, AsyncMockLLM):
                self.is_async = True
        else:
            raise ValueError("Model does not have a valid LLM interface")

        self.model = model
        self.relative_grade_template = relative_grade_template

    def _get_conversation_prompt(self, messages: List[Dict[str, str]], model_type: str = "mistral"):
        conv = get_conv_template(model_type)

        for message in messages:
            if message["role"] == "system":
                conv.set_system_message(message["content"])
            elif message["role"] == "user":
                conv.append_message(conv.roles[0], message["content"])

        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        return prompt


    def _check_inputs(self, instructions, responses, rubric, reference_answers):
        if len(instructions) != len(responses):
            raise ValueError(
                "Length of instructions must match the length of responses"
            )

        # If rubric is a list, check its length matches the length of instructions
        if isinstance(rubric, list) and len(rubric) != len(instructions):
            raise ValueError("Length of rubric must match the length of instructions")
        elif isinstance(rubric, list) and len(rubric) == len(instructions):
            pass
        elif isinstance(rubric, str):
            rubric = [rubric] * len(
                instructions
            )  # Apply the same rubric to all if it's not a list
        else:
            raise ValueError("Rubric must be a string or a list of strings")

        # Handle reference answers
        if isinstance(reference_answers, list) and len(reference_answers) != len(
            instructions
        ):
            raise ValueError(
                "Length of reference answers must match the length of instructions"
            )
        elif isinstance(reference_answers, list):
            pass
        else:
            warnings.warn(
                "Reference answer was not provided. This may result in suboptimal grading performance. Consider providing a reference answer for best results."
            )
            reference_answers = [None] * len(
                instructions
            )  # Default to None if not provided

        return instructions, responses, rubric, reference_answers


    def relative_grade_preference(
        self,
        *,
        instructions: List[str],
        responses_A: List[str],
        responses_B: List[str],
        rubric: List[str] | str,
        reference_answers: List[str] = None,
        params: Dict[str, Any] = {},
        model_name: str,
    ) -> Tuple[List[str], List[int]]:
        """
        Grades a batch of responses relatively based on the provided instructions and paired responses.

        :param instructions: List of instructions for each paired responses.
        :param responses_A: List of first responses in each pair.
        :param responses_B: List of second responses in each pair.
        :param params: Additional parameters for the model completion requests. Refer to the vllm SamplingParmas class.
        :return: A tuple containing lists of feedbacks and scores.
        """

        instructions, _, rubric, reference_answers = self._check_inputs(
            instructions, list(zip(responses_A, responses_B)), rubric, reference_answers
        )

        inputs = []
        for idx, (instruction, response_a, response_b) in enumerate(
            zip(instructions, responses_A, responses_B)
        ):
            rubric_ = rubric[idx]
            reference_answer = reference_answers[idx]
            content = self.relative_grade_template.format(
                instruction=instruction,
                response_A=response_a,
                response_B=response_b,
                rubric=rubric_,
                reference_answer=reference_answer,
            )
            messages = [
                {"role": "system", "content": REL_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ]
            if hasattr(self.model, "validate_vllm"):
                input_ = self._get_conversation_prompt(messages, model_type=model_name)
            else:
                input_ = messages
            inputs.append(input_)

        if self.is_async:
            feedbacks, scores = asyncio.run(
                async_batch_completions_with_retries(
                    self.model,
                    inputs,
                    mode="relative",
                    params=params,
                )
            )
        else:
            feedbacks, scores = batch_completions_with_retries(
                self.model,
                inputs,
                mode="relative",
                params=params,
            )

        return feedbacks, scores
    
    def relative_grade(
        self,
        *,
        instructions: List[str],
        responses_A: List[str],
        responses_B: List[str],
        params: Dict[str, Any] = {},
        mode: str = None,
        model_name: str,
    ) -> Tuple[List[str], List[int]]:
        """
        Grades a batch of responses relatively based on the provided instructions and paired responses.

        :param instructions: List of instructions for each paired responses.
        :param responses_A: List of first responses in each pair.
        :param responses_B: List of second responses in each pair.
        :param params: Additional parameters for the model completion requests. Refer to the vllm SamplingParmas class.
        :return: A tuple containing lists of feedbacks and scores.
        """


        inputs = []
        for idx, (instruction, response_a, response_b) in enumerate(
            zip(instructions, responses_A, responses_B)
        ):
            # rubric_ = rubric[idx]
            # reference_answer = reference_answers[idx]
            content = self.relative_grade_template.format(
                instruction=instruction,
                response_A=response_a,
                response_B=response_b,
            )
            # print('content:', content)
            messages = [
                {"role": "system", "content": REL_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ]
            if hasattr(self.model, "validate_vllm"):
                input_ = self._get_conversation_prompt(messages, model_type=model_name)
            else:
                input_ = messages
            inputs.append(input_)

        if self.is_async:
            print('mode:', mode)
            feedbacks, scores = asyncio.run(
                async_batch_completions_with_retries(
                    self.model,
                    inputs,
                    mode=mode,
                    params=params,
                )
            )
        elif mode == "Auto-J":
            feedbacks, scores = batch_completions_with_retries(
                self.model,
                inputs,
                mode="Auto-J",
                params=params,
            )
        elif mode == "no_parse":
            feedbacks, scores = batch_completions_with_retries(
                self.model,
                inputs,
                mode="no_parse",
                params=params,
            )
        else:
            feedbacks, scores = batch_completions_with_retries(
                self.model,
                inputs,
                mode="relative",
                params=params,
            )

        return feedbacks, scores
    