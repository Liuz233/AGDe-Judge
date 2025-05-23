from typing import List, Union


class VLLMError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        super().__init__(self.message)


try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
except ImportError as e:
    raise VLLMError(
        status_code=1,
        message="Failed to import 'vllm' package. Make sure it is installed correctly.",
    ) from e


class VLLM:
    def __init__(
        self,
        model: str,
        lora_config: dict,
        **vllm_kwargs,
    ) -> None:
        self.model: str = model
        self.lora_config: dict = lora_config
        print(f'lora_config: {lora_config}')
        self.model: LLM = LLM(
            model=self.model,
            **vllm_kwargs,
        )

    def validate_vllm(self):
        return True

    def completions(
        self,
        prompts: List[str],
        use_tqdm: bool = True,
        **kwargs: Union[int, float, str],
    ) -> List[str]:
        prompts = [prompt.strip() for prompt in prompts]
        params = SamplingParams(**kwargs)
        if self.lora_config:
            outputs = self.model.generate(prompts, params, use_tqdm=use_tqdm, lora_request=LoRARequest(self.lora_config["name"], self.lora_config["id"], self.lora_config["path"]))
        else:
            outputs = self.model.generate(prompts, params, use_tqdm=use_tqdm)
        outputs = [output.outputs[0].text for output in outputs]
        return outputs
