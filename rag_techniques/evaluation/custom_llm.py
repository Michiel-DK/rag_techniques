from transformers import AutoModelForCausalLM, AutoTokenizer
from deepeval.models.base_model import DeepEvalBaseLLM
from mlx_lm import load, generate


class Mistral7B(DeepEvalBaseLLM):
    def __init__(
        self,
        model,
        tokenizer
    ):
        self.model = model
        self.tokenizer = tokenizer

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        model = self.load_model()

        device = "mps" # the device to load the model onto

        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(device)
        model.to(device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
        return self.tokenizer.batch_decode(generated_ids)[0]

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "Mistral 7B"


import transformers
import torch
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from deepeval.models import DeepEvalBaseLLM


class CustomLlama3_8B(DeepEvalBaseLLM):
    def __init__(self):
        # quantization_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_compute_dtype=torch.float16,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_use_double_quant=True,
        # )

        # model_4bit = AutoModelForCausalLM.from_pretrained(
        #     "mlx-community/Meta-Llama-3-8B-Instruct-4bit",
        #     device_map="auto",
        #    # quantization_config=quantization_config,
        # )
        # tokenizer = AutoTokenizer.from_pretrained(
        #     "mlx-community/Meta-Llama-3-8B-Instruct-4bits"
        # )
        
        model_4bit, tokenizer = load("mlx-community/Meta-Llama-3-8B-Instruct-4bit")

        self.model = model_4bit
        self.tokenizer = tokenizer

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        model = self.load_model()

        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            use_cache=True,
            device_map="auto",
            max_length=2500,
            do_sample=True,
            top_k=5,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        return pipeline(prompt)

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "Llama-3 8B"
    

if __name__ == '__main__':
    #path = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    # path = "mistralai/Mistral-7B-v0.1"
    # model = AutoModelForCausalLM.from_pretrained(path)
    # tokenizer = AutoTokenizer.from_pretrained(path)

    custom_llm = CustomLlama3_8B()
    print(custom_llm.generate("Write me a joke about cycling"))
    
    