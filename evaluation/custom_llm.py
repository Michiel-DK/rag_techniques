from transformers import AutoModelForCausalLM, AutoTokenizer
from deepeval.models.base_model import DeepEvalBaseLLM

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
    

if __name__ == '__main__':
    #path = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    path = "mistralai/Mistral-7B-v0.1"
    model = AutoModelForCausalLM.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)

    mistral_7b = Mistral7B(model=model, tokenizer=tokenizer)
    print(mistral_7b.generate("Write me a joke about cycling"))
    
    