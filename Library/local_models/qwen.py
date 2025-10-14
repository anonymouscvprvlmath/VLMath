from typing import Optional
from Library.local_models.local_model import LocalModel
from Library.prompts import context_maker
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer
import torch
import json
from datasets import load_dataset, load_from_disk
import copy

class Qwen(LocalModel):

    def __init__(self, base_model="Qwen/Qwen2.5-VL-3B-Instruct", fine_tuned_path=""):
        super().__init__(base_model, fine_tuned_path)
        # print("Special tokens:", self.tokenizer.special_tokens_map)
    
    def init_model(self):
        model_path = self.fine_tuned_path or self.base_model
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto"
        )


    def init_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self.base_model,
            trust_remote_code=True
        )

    def init_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            trust_remote_code=True
        )

    def init_special_tokens(self):
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        print(self.tokenizer.special_tokens_map)
        # self.end_token_id = self.tokenizer.encode("<|end|>", add_special_tokens=False)
        # self.assistant_token_id = self.tokenizer.encode("<|assistant|>", add_special_tokens=False)

    def _prepare_for_inference(self):
        if not torch.cuda.is_available():
            raise EnvironmentError("CUDA isn't available, won't inference on cpu")

        self.device = torch.device("cuda")
        self.model.to(self.device)
        self.prepared_for_inference = True

    def run_inference(
            self, 
            prompt: str = "", 
            images: Optional[list] = None, 
            messages: Optional[list] = None,
            system: Optional[dict] = None
        ):
        system = system or []
        images = images or []
        messages = messages or [] 
        if not self.prepared_for_inference:
            self._prepare_for_inference()

        input_messages = [system]

        for message in messages:
            input_messages.append(
                {"role": message["role"], "content": [
                    {"type": "text", "text": message["content"]}
                ]})
            

        # Overriding the input_messages if prompt exists
        if prompt:
            input_messages = [{"role": "user", 
                               "content": [
                                    {"type": "text", "text": prompt}
                               ] 
                               }
                            ]


        inputs = self.processor.apply_chat_template(
            input_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens = 1000,
            )

        response = self.processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        return response
    
    def process_dataset(self, dataset_name: str, split:str, preprocess: bool):

        if preprocess:
            dataset = load_dataset(dataset_name, split=split)
            with open("./MetaData/mathvision_v2.json") as f:
                self.conv_list= json.load(f)

            original_column_names = dataset.column_names
            processed_dataset = dataset.map(self.process_math_vision)
            print(processed_dataset[0])
            indicies_to_keep = [i for i, flag in enumerate(processed_dataset["drop"])if not flag]
            processed_dataset = processed_dataset.select(indicies_to_keep)
            print("Dataset prorcessed...")
            processed_dataset = processed_dataset.remove_columns(original_column_names + ["drop"])

            print("Saving the processed dataset to disk")
            processed_dataset.save_to_disk(dataset_name + "_processed")
            return dataset 

        else:
            return load_from_disk(dataset + "_processed")

    def process_math_vision(self, example):
        image = example["decoded_image"]
        ex_id = int(example["id"])
        try:
            conversations = self.conv_list[ex_id]["messages"]
        except KeyError as e:
            return {
                "input_ids": [],
                "attention_mask": [],
                "pixel_values": [],  # If this is a list of tensors, ensure the placeholder matches the expected type
                "image_grid_thw": [],   # Add any other keys your successful examples return
                "labels": [],
                "drop": True # Flag to be filtered out 
            }
            print

        context = context_maker(example)
        messages = [{"role": "system", "content": [
            {"type": "text", "text": context},
            {"type": "image", "image": image}
        ]}]

        for item in conversations:
            if "teacher" in item:
                messages.append({"role": "assistant", "content": [
                    {"type": "text", "text": item["teacher"]}
                ]})

            elif "student" in item:
                messages.append({"role": "user", "content": [
                    {"type": "text", "text": item["student"]}
                ]})

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    
        inputs["input_ids"] = inputs["input_ids"][0].tolist()
        inputs["labels"] = copy.deepcopy(inputs["input_ids"])
        inputs["attention_mask"] = inputs["attention_mask"][0].tolist()
        inputs["image_grid_thw"] = inputs["image_grid_thw"][0].tolist()

        turn_to_neg = True
        assistant_start = self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        assistant_end = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)

        # print(assistant_start, assistant_end)
        # print(self.tokenizer.encode("<|end|>"))
        # print(self.tokenizer.encode("<|assistant|>"))
        # print(self.assistant_token_id)
        # print(self.end_token_id)
        # print(inputs["input_ids"])

        labels = inputs["labels"]
        # print(type(labels))
        # print(labels)
        turn_to_neg = True
        i = 0
        while i < len(labels):
            # Check for assistant start pattern
            if labels[i:i+len(assistant_start)] == assistant_start:
                turn_to_neg = False
                i += len(assistant_start)
                continue

            # Check for end of block
            if labels[i:i+len(assistant_end)] == assistant_end:
                if turn_to_neg:
                    labels[i] = -100

                turn_to_neg = True
                i += len(assistant_end)
                continue

            # Mask if outside assistant block
            if turn_to_neg:
                labels[i] = -100

            i += 1
        # print(self.tokenizer.decode(inputs["input_ids"]))
        to_decode = [val for val in inputs["labels"] if val != -100]
        # print(self.tokenizer.decode(to_decode))
        # for idx, val in enumerate(inputs["labels"]):
        #     if val == self.assistant_token_id:
        #         turn_to_neg = False
        #     elif val == self.end_token_id:
        #         turn_to_neg = True
        #         continue

        #     if turn_to_neg:
        #         inputs["labels"][idx] = -100

        inputs["drop"] = False

        # print("\n===== PROCESSOR OUTPUT =====")
        # for k, v in inputs.items():
        #     if hasattr(v, "shape"):
        #         print(f"{k}: shape={tuple(v.shape)} dtype={getattr(v, 'dtype', None)}")
        #     else:
        #         print(f"{k}: type={type(v), len(v)}")
        # print("============================\n")

        # print(inputs["labels"])


        return inputs


    # def process_math_vision(self, example):
    #     end_token_id = self.tokenizer.encode("<|end|>")[1]
    #     assistant_token_id = self.tokenizer.encode("<|assistant|>")[1]
    #     image = example["decoded_image"]

    #     ex_id = int(example['id'])

    #     try:
    #         convs = self.conv_list[ex_id]["messages"]
    #     except KeyError as e:
    #         return {
    #             "input_ids": [],
    #             "attention_mask": [],
    #             "pixel_values": [],  # If this is a list of tensors, ensure the placeholder matches the expected type
    #             "image_sizes": [],   # Add any other keys your successful examples return
    #             "labels": [],
    #             "drop": True # Flag to be filtered out 
    #         }

    #     context_text = f"""
    #     Here is the context for this problem: 
    #     Question:
    #     {example['question']}

    #     Question image:

    #     <|image_1|>\n

    #     Correct answer:
    #     {example['answer']}
        
    #     Teacher student conversation:
    #     """

    #     messages = [{"role": "user", "content": context_text}]

    #     for item in convs:
    #         if "teacher" in item:
    #             messages.append({"role": "assistant", "content": item["teacher"]})

    #         elif "student" in item:
    #             messages.append({"role": "user", "content": item["student"]})

    #     full_prompt = self.tokenizer.apply_chat_template(
    #         messages,
    #         tokenize=False,
    #         add_generation_prompt=False
    #     )

    #     model_inputs = self.processor(
    #         text=full_prompt,
    #         images=image
    #     )
    #     print(model_inputs.keys)

    #     input_ids = model_inputs["input_ids"]
    #     input_ids[input_ids == -1] = self.tokenizer.pad_token_id
    #     model_inputs["input_ids"] = input_ids[0].tolist()
    #     model_inputs["labels"] = input_ids[0].tolist()
    #     model_inputs["attention_mask"] = model_inputs["attention_mask"][0]

    #     model_inputs["image_grid_thw"] = model_inputs.get("image_grid_thw", [(1, 14, 14)])

    #     turn_to_neg = True
    #     for idx, val in enumerate(model_inputs["labels"]):
    #         if val == assistant_token_id:
    #             turn_to_neg = False
    #         elif val == end_token_id:
    #             turn_to_neg = True
    #             continue

    #         if turn_to_neg:
    #             model_inputs["labels"][idx] = -100
    #         model_inputs["drop"] = False

    #     return model_inputs
