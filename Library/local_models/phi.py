
from typing import Optional
from Library.local_models.local_model import LocalModel
from transformers import AutoModelForCausalLM, AutoProcessor
import torch


class Phi(LocalModel):
    
    def __init__(self, base_model="microsoft/Phi-3.5-vision-instruct", fine_tuned_path=""):
        super().__init__(base_model, fine_tuned_path)

    def init_model(self):
        model_path = self.fine_tuned_path or self.base_model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

    def init_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self.base_model,
            trust_remote_code=True
        )

    def init_tokenizer(self):
        self.tokenizer = self.processor.tokenizer

    def init_special_tokens(self):
        self.eos_token = "<|end|>"
        self.pad_token = self.eos_token

        self.tokenizer.eos_token = self.eos_token
        self.tokenizer.pad_token = self.eos_token

        self.model.config.eos_token_id = self.tokenizer.convert_tokens_to_ids(self.eos_token)
        self.model.config.pad_token_id = self.model.config.eos_token_id

    def _prepare_for_inference(self):
        if not torch.cuda.is_available():
            raise EnvironmentError("CUDA isn't available, won't inference on cpu")

        self.device = torch.device("cuda")
        print(f"Using {self.device}")
        self.model.to(self.device)
        self.prepared_for_inference = True

    def run_inference(
            self, 
            prompt: str = "", 
            images: Optional[list] = None, 
            messages: Optional[list] = None
        ):

        images = images or []
        messages = messages or [] 
        if not self.prepared_for_inference:
            self._prepare_for_inference()

        image_str_lst = [f"<|image_{i}|>" for i in range(len(images))]
        image_str = "\n".join(image_str_lst)

        if not messages:
            messages = messages = [
                {"role": "user", "content": prompt + image_str}
            ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            prompt,
            images if len(images) > 0 else None
        ).to(self.device)

        with torch.inference_mode():
            # Using KV Cache for generation
            output = self.model.generate(
                **inputs,
                max_new_tokens = 200,
                eos_token_id = self.model.config.eos_token_id,
                pad_token_id = self.model.config.eos_token_id,
            )

        output = output[:, inputs["input_ids"].shape[1] :]
        clean_ids = output.masked_fill(output == -1, self.tokenizer.eos_token_id)
        response = self.tokenizer.decode(clean_ids[0], skip_special_tokens = True)

        return response