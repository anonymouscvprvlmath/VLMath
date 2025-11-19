from typing import Optional, Tuple
from Library.local_models.local_model import LocalModel
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
import torch


class Phi(LocalModel):

    def __init__(
        self,
        base_model="microsoft/Phi-3.5-vision-instruct",
        fine_tuned_path="",
        quantize_4bit: bool = False,
        quantize_8bit: bool = False,
    ):
        self.quantize_4bit = quantize_4bit
        self.quantize_8bit = quantize_8bit
        super().__init__(base_model, fine_tuned_path)

    def init_model(self):
        model_path = self.fine_tuned_path or self.base_model
        bnb_4_bit_config = None
        if self.quantize_4bit:
            bnb_4_bit_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        if self.quantize_8bit:
            bnb_8_bit_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_has_fp16_weight=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            quantization_config=(
                bnb_4_bit_config
                if self.quantize_4bit
                else (bnb_8_bit_config if self.quantize_8bit else None)
            ),
        )

    def init_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self.base_model, trust_remote_code=True
        )

    def init_tokenizer(self):
        self.tokenizer = self.processor.tokenizer

    def init_special_tokens(self):
        self.eos_token = "<|end|>"
        self.pad_token = self.eos_token

        self.tokenizer.eos_token = self.eos_token
        self.tokenizer.pad_token = self.eos_token

        self.model.config.eos_token_id = self.tokenizer.convert_tokens_to_ids(
            self.eos_token
        )
        self.model.config.pad_token_id = self.model.config.eos_token_id

    def _prepare_for_inference(self):
        if not torch.cuda.is_available():
            raise EnvironmentError("CUDA isn't available, won't inference on cpu")

        self.device = torch.device("cuda")
        print(f"Using {self.device}")
        if not (self.quantize_4bit or self.quantize_8bit):
            self.model.to(self.device)
        self.prepared_for_inference = True
        self.print_model_quantization_info(model=self.model)

    def run_inference(
        self,
        prompt: str = "",
        images: Optional[list] = None,
        messages: Optional[list] = None,
    ):

        images = images or []
        messages = messages or []
        if not self.prepared_for_inference:
            self._prepare_for_inference()

        image_str_lst = [f"<|image_{i}|>" for i in range(len(images))]
        image_str = "\n".join(image_str_lst)

        if not messages:
            messages = messages = [{"role": "user", "content": prompt + image_str}]

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(prompt, images if len(images) > 0 else None).to(
            self.device
        )

        with torch.inference_mode():
            # Using KV Cache for generation
            output = self.model.generate(
                **inputs,
                max_new_tokens=500,
                eos_token_id=self.model.config.eos_token_id,
                pad_token_id=self.model.config.eos_token_id,
            )

        output = output[:, inputs["input_ids"].shape[1] :]
        clean_ids = output.masked_fill(output == -1, self.tokenizer.eos_token_id)
        response = self.tokenizer.decode(clean_ids[0], skip_special_tokens=True)

        return response

    def process_math_vision(self, example):
        image = example["decoded_image"]

        ex_id = example["id"]
        try:
            convs = self.conv_dict[ex_id][f"{self.subcategory}_conv"]
        except KeyError as e:
            return {
                "input_ids": [],
                "attention_mask": [],
                "pixel_values": [],
                "image_sizes": [],
                "labels": [],
                "drop": True,  # Flag to be filtered out
            }

        context_text = f"""
        Here is the context for this problem: 
        Question:
        {example['question']}

        Question image:

        <|image_1|>\n

        Correct answer:
        {example['answer']}
        
        Teacher student conversation:
        """

        messages = [{"role": "user", "content": context_text}]

        for item in convs:

            if "teacher" in item:
                messages.append({"role": "assistant", "content": item["teacher"]})

            elif "student" in item:
                messages.append({"role": "user", "content": item["student"]})

        full_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        model_inputs = self.processor(text=full_prompt, images=image)

        input_ids = model_inputs["input_ids"]
        input_ids[input_ids == -1] = self.processor.tokenizer.pad_token_id
        model_inputs["input_ids"] = input_ids[0].tolist()
        model_inputs["labels"] = input_ids[0].tolist()
        model_inputs["attention_mask"] = model_inputs["attention_mask"][0]

        model_inputs["drop"] = False

        return model_inputs

    def count_inference_token(
        self,
        prompt: str = "",
        images: Optional[list] = None,
        messages: Optional[list] = None,
    ) -> Tuple[str, int]:

        images = images or []
        messages = messages or []
        if not self.prepared_for_inference:
            self._prepare_for_inference()

        image_str_lst = [f"<|image_{i}|>" for i in range(len(images))]
        image_str = "\n".join(image_str_lst)

        if not messages:
            messages = messages = [{"role": "user", "content": prompt + image_str}]

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(prompt, images if len(images) > 0 else None).to(
            self.device
        )

        with torch.inference_mode():
            # Using KV Cache for generation
            output = self.model.generate(
                **inputs,
                max_new_tokens=500,
                eos_token_id=self.model.config.eos_token_id,
                pad_token_id=self.model.config.eos_token_id,
            )

        output = output[:, inputs["input_ids"].shape[1] :]
        clean_ids = output.masked_fill(output == -1, self.tokenizer.eos_token_id)
        response = self.tokenizer.decode(clean_ids[0], skip_special_tokens=True)
        return response, len(clean_ids[0])
