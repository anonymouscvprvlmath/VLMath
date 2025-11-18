from typing import Optional
from Library.local_models.local_model import LocalModel
from Library.prompts import context_maker
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
)
import torch
import json
from datasets import load_dataset, load_from_disk
import copy


class Qwen(LocalModel):

    def __init__(
        self,
        base_model="Qwen/Qwen2.5-VL-3B-Instruct",
        fine_tuned_path="",
        quantize_4bit=False,
        quantize_8bit=False,
    ):
        super().__init__(base_model, fine_tuned_path)
        # print("Special tokens:", self.tokenizer.special_tokens_map)

    def init_model(self):
        model_path = self.fine_tuned_path or self.base_model
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto"
        )
        self.model.gradient_checkpointing_enable()
        print(next(self.model.parameters()).dtype)

    def init_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self.base_model, trust_remote_code=True
        )

    def init_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model, trust_remote_code=True
        )

    def init_special_tokens(self):
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        # print(self.tokenizer.special_tokens_map)
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
        system: Optional[dict] = None,
    ):
        system = system or []
        images = images or []
        messages = messages or []
        if not self.prepared_for_inference:
            self._prepare_for_inference()

        input_messages = [system]

        for message in messages:
            input_messages.append(
                {
                    "role": message["role"],
                    "content": [{"type": "text", "text": message["content"]}],
                }
            )

        # Overriding the input_messages if prompt exists
        if prompt:
            input_messages = [
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
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
                max_new_tokens=1000,
            )

        response = self.processor.decode(
            outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        )
        return response

    # def process_dataset(self, dataset_name: str, preprocess: bool, args):

    #     if preprocess:
    #         dataset = load_dataset(dataset_name, split=args.dataset_split)
    #         with open("./MetaData/mathvision_v2.json") as f:
    #             self.conv_list = json.load(f)

    #         original_column_names = dataset.column_names
    #         processed_dataset = dataset.map(
    #             self.process_math_vision, writer_batch_size=50
    #         )
    #         # print(processed_dataset[0])
    #         print("Selecting the non drop ones")
    #         indicies_to_keep = [
    #             i for i, flag in enumerate(processed_dataset["drop"]) if not flag
    #         ]
    #         processed_dataset = processed_dataset.select(indicies_to_keep)
    #         print("Dataset prorcessed...")
    #         processed_dataset = processed_dataset.remove_columns(
    #             original_column_names + ["drop"]
    #         )

    #         print("Saving the processed dataset to disk")
    #         processed_dataset.save_to_disk(
    #             dataset_name + f"_processed_with_{self.base_model}"
    #         )
    #         return processed_dataset

    #     else:
    #         return load_from_disk(dataset_name + f"_processed_with_{self.base_model}")

    def process_math_vision(self, example):
        image = example["decoded_image"]
        ex_id = example["id"]
        try:
            # conversations = self.conv_dict[ex_id]["messages"]
            conversations = self.conv_dict[ex_id][f"{self.subcategory}_conv"]
        except KeyError as e:
            return {
                "input_ids": [],
                "attention_mask": [],
                "pixel_values": [],
                "image_grid_thw": [],
                "labels": [],
                "drop": True,  # Flag to be filtered out
            }

        # context = f"""
        # Here is the context for this problem:
        # Question:
        # {example['question']}

        # Question image:

        # <|image_1|>\n

        # Correct answer:
        # {example['answer']}

        # Teacher student conversation:
        # """

        # messages = [
        #     {
        #         # "role": "system",
        #         "role": "user",
        #         "content": [
        #             {"type": "text", "text": context},
        #             {"type": "image", "image": image},
        #         ],
        #     }
        # ]
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a patient math teacher who helps the student reason step by step.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Question:\n{example['question']}"},
                    {"type": "image", "image": example["decoded_image"]},
                    {
                        "type": "text",
                        "text": f"\nCorrect answer: {example['answer']}\nLet's begin the discussion.",
                    },
                ],
            },
        ]

        for item in conversations:
            if "teacher" in item:
                messages.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": item["teacher"]}],
                    }
                )

            elif "student" in item:
                messages.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": item["student"]}],
                    }
                )

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs["input_ids"] = inputs["input_ids"].flatten()
        input_ids = inputs["input_ids"]
        inputs["labels"] = copy.deepcopy(input_ids)
        inputs["attention_mask"] = inputs["attention_mask"].flatten()
        inputs["image_grid_thw"] = inputs["image_grid_thw"].flatten()

        labels = inputs["labels"]
        assistant_start = torch.tensor(
            self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False),
            device=labels.device,
        )
        assistant_end = torch.tensor(
            self.tokenizer.encode("<|im_end|>", add_special_tokens=False),
            device=labels.device,
        )
        turn_to_neg = True
        i = 0

        while i < labels.numel():
            # start of assistant
            if i + len(assistant_start) <= labels.numel() and torch.equal(
                labels[i : i + len(assistant_start)], assistant_start
            ):
                turn_to_neg = False
                i += len(assistant_start)
                continue

            # end of assistant
            if i + len(assistant_end) <= labels.numel() and torch.equal(
                labels[i : i + len(assistant_end)], assistant_end
            ):
                if turn_to_neg:
                    labels[i] = -100

                turn_to_neg = True
                i += len(assistant_end)
                continue

            if turn_to_neg:
                labels[i] = -100
            i += 1

        inputs["labels"] = copy.deepcopy(labels)
        inputs["drop"] = False
        # self.debug_print_conversation_tokens(inputs)
        self.debug_print_trainable_text(inputs)
        return inputs

    # def process_math_vision(self, example):
    #     image = example["decoded_image"]
    #     ex_id = example["id"]
    #     try:
    #         # conversations = self.conv_dict[ex_id]["messages"]
    #         conversations = self.conv_dict[ex_id][f"{self.subcategory}_conv"]
    #     except KeyError as e:
    #         return {
    #             "input_ids": [],
    #             "attention_mask": [],
    #             "pixel_values": [],
    #             "image_grid_thw": [],
    #             "labels": [],
    #             "drop": True,  # Flag to be filtered out
    #         }

    #     # context = f"""
    #     # Here is the context for this problem:
    #     # Question:
    #     # {example['question']}

    #     # Correct answer:
    #     # {example['answer']}

    #     # Teacher student conversation:
    #     # """

    #     # messages = [
    #     #     {
    #     #         "role": "system",
    #     #         # "role": "user",
    #     #         "content": [
    #     #             {"type": "text", "text": context},
    #     #             {"type": "image", "image": image},
    #     #         ],
    #     #     }
    #     # ]

    #     messages = [
    #         {
    #             "role": "system",
    #             "content": [
    #                 {
    #                     "type": "text",
    #                     "text": "You are a patient math teacher who helps the student reason step by step.",
    #                 }
    #             ],
    #         },
    #         {
    #             "role": "user",
    #             "content": [
    #                 {"type": "text", "text": f"Question:\n{example['question']}"},
    #                 {"type": "image", "image": example["decoded_image"]},
    #                 {
    #                     "type": "text",
    #                     "text": f"\nCorrect answer: {example['answer']}\nLet's begin the discussion.",
    #                 },
    #             ],
    #         },
    #     ]

    #     for item in conversations:
    #         if "teacher" in item:
    #             # print(item["teacher"])
    #             messages.append(
    #                 {
    #                     "role": "assistant",
    #                     "content": [{"type": "text", "text": item["teacher"]}],
    #                 }
    #             )

    #         elif "student" in item:
    #             messages.append(
    #                 {
    #                     "role": "user",
    #                     "content": [{"type": "text", "text": item["student"]}],
    #                 }
    #             )

    #     inputs = self.processor.apply_chat_template(
    #         messages,
    #         tokenize=True,
    #         return_dict=True,
    #         return_tensors="pt",
    #         return_assistant_tokens_mask=True,
    #     )

    #     # print(inputs.keys())

    #     inputs["input_ids"] = inputs["input_ids"].flatten()
    #     # input_ids = inputs["input_ids"]
    #     # inputs["labels"] = copy.deepcopy(input_ids)
    #     labels = inputs["input_ids"].clone()
    #     labels[~inputs["assistant_masks"].flatten()] = -100
    #     inputs["attention_mask"] = inputs["attention_mask"].flatten()
    #     inputs["image_grid_thw"] = inputs["image_grid_thw"].flatten()
    #     inputs["labels"] = labels
    #     inputs["drop"] = False
    #     # labels = inputs["labels"]
    #     # self.debug_print_conversation_tokens(inputs)

    def debug_print_trainable_text(self, inputs):
        """
        Prints only the text segments that are *trainable* (labels != -100).
        This shows exactly what the loss is computed on.
        """
        tokenizer = self.tokenizer
        input_ids = inputs["input_ids"].flatten()
        labels = inputs["labels"].flatten()

        # Identify contiguous regions of trainable tokens
        trainable_indices = (labels != -100).nonzero(as_tuple=True)[0].tolist()
        if not trainable_indices:
            print("⚠️ No trainable tokens in this example.")
            return

        # Group contiguous indices into spans
        spans = []
        start = trainable_indices[0]
        prev = start
        for idx in trainable_indices[1:]:
            if idx != prev + 1:
                spans.append((start, prev))
                start = idx
            prev = idx
        spans.append((start, prev))

        print("\n=== TRAINABLE (LOSS) REGIONS ===")
        for s, e in spans:
            decoded = tokenizer.decode(
                input_ids[s : e + 1].tolist(), skip_special_tokens=False
            )
            # small cleanup for readability
            decoded = decoded.replace("\n", "\\n")
            print(f"\nTokens {s}-{e} ({e-s+1} tokens):")
            print(decoded[:800])
        print("=================================\n")

        total = len(input_ids)
        trainable = len(trainable_indices)
        print(f"Trainable tokens: {trainable}/{total} ({trainable/total*100:.2f}%)")

    # VIBE
    def debug_print_conversation_tokens(self, inputs):
        """
        Pretty-prints a tokenized Qwen chat example, showing which tokens belong
        to the system, user, and assistant, and which are trainable.
        """

        tokenizer = self.tokenizer
        input_ids = inputs["input_ids"].flatten().tolist()
        labels = inputs.get("labels", None)
        if labels is not None:
            labels = labels.flatten().tolist()

        decoded = tokenizer.decode(input_ids, skip_special_tokens=False)
        print("\n=== FULL DECODED PROMPT ===")
        print(decoded[:2000])  # print first ~2000 chars
        print("===========================\n")

        start_sys = tokenizer.encode("<|im_start|>system", add_special_tokens=False)
        start_usr = tokenizer.encode("<|im_start|>user", add_special_tokens=False)
        start_ast = tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
        end_tok = tokenizer.encode("<|im_end|>", add_special_tokens=False)

        i = 0
        role = None
        snippets = []
        while i < len(input_ids):
            if (
                i + len(start_sys) <= len(input_ids)
                and input_ids[i : i + len(start_sys)] == start_sys
            ):
                role = "system"
                i += len(start_sys)
                continue
            if (
                i + len(start_usr) <= len(input_ids)
                and input_ids[i : i + len(start_usr)] == start_usr
            ):
                role = "user"
                i += len(start_usr)
                continue
            if (
                i + len(start_ast) <= len(input_ids)
                and input_ids[i : i + len(start_ast)] == start_ast
            ):
                role = "assistant"
                i += len(start_ast)
                continue
            if (
                i + len(end_tok) <= len(input_ids)
                and input_ids[i : i + len(end_tok)] == end_tok
            ):
                if snippets:
                    text = tokenizer.decode(snippets, skip_special_tokens=False)
                    print(
                        f"🟦 {role.upper()} ({len(snippets)} tokens):\n{text[:600]}\n"
                    )
                    snippets = []
                role = None
                i += len(end_tok)
                continue
            if role:
                snippets.append(input_ids[i])
            i += 1

        if labels is not None:
            num_trainable = sum(1 for l in labels if l != -100)
            print(
                f"Trainable tokens: {num_trainable}/{len(labels)} "
                f"({num_trainable/len(labels)*100:.2f}%)"
            )


# VIBE CODE
def debug_print_inputs(inputs):
    print("\n=== Debug: process_math_vision return structure ===")
    for k, v in inputs.items():
        # Detect type and shape info
        if hasattr(v, "shape"):
            print(f"{k:20s} | type: {type(v)} | shape: {tuple(v.shape)}")
        elif isinstance(v, (list, tuple)):
            if len(v) > 0 and hasattr(v[0], "shape"):
                print(
                    f"{k:20s} | type: list[{type(v[0])}] | first shape: {tuple(v[0].shape)} | len: {len(v)}"
                )
            elif len(v) > 0 and isinstance(v[0], (list, tuple)):
                print(
                    f"{k:20s} | type: nested list | first lens: {[len(x) for x in v[:3]]} ... | len: {len(v)}"
                )
            else:
                print(f"{k:20s} | type: list | len: {len(v)} | sample: {str(v[:3])}")
        else:
            print(f"{k:20s} | type: {type(v)} | value: {str(v)[:80]}")
    print("====================================================\n")
