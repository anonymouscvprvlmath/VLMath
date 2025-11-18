from abc import ABC, abstractmethod
from typing import Optional
from transformers import PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin
import torch
import json
from datasets import load_dataset, load_from_disk


class LocalModel(ABC):

    model: Optional[PreTrainedModel]
    processor: Optional[ProcessorMixin]
    tokenizer: Optional[PreTrainedTokenizerBase]
    base_model: str
    fine_tuned_path: Optional[str]
    eos_token: str
    pad_token: str
    prepared_for_inference: bool = False

    def __init__(self, base_model, fine_tuned_path=""):
        self.base_model = base_model
        self.fine_tuned_path = fine_tuned_path if fine_tuned_path else None
        self.model = None
        self.init_model()
        self.init_processor()
        self.init_tokenizer()
        self.init_special_tokens()

    @abstractmethod
    def init_model(self): ...

    @abstractmethod
    def init_processor(self): ...

    @abstractmethod
    def init_tokenizer(self): ...

    @abstractmethod
    def init_special_tokens(self): ...

    @abstractmethod
    def run_inference(self, text: str, images: list, messages: list): ...

    def process_dataset(self, dataset_name: str, preprocess: bool, args):

        self.subcategory = "scaffolding"
        if preprocess:
            dataset = load_dataset(dataset_name, split=args.dataset_split)
            with open(f"./MetaData/mathvision_{self.subcategory}.json") as f:
                self.conv_dict = json.load(f)

            original_column_names = dataset.column_names
            processed_dataset = dataset.map(
                self.process_math_vision, writer_batch_size=50
            )
            print("Selecting the non drop ones")
            indicies_to_keep = [
                i for i, flag in enumerate(processed_dataset["drop"]) if not flag
            ]
            processed_dataset = processed_dataset.select(indicies_to_keep)
            print("Dataset prorcessed...")
            processed_dataset = processed_dataset.remove_columns(
                original_column_names + ["drop"]
            )

            print("Saving the processed dataset to disk")
            processed_dataset.save_to_disk(
                dataset_name + f"_processed_with_{self.base_model}_{self.subcategory}"
            )
            return processed_dataset

        else:
            return load_from_disk(
                dataset_name + f"_processed_with_{self.base_model}_{self.subcategory}"
            )

    # GPT VIBE
    def print_model_quantization_info(self, model, label="Model"):
        """Prints a structured summary of quantization and dtype per model section."""
        print(f"\n{'='*100}")
        print(f"{label} Quantization Summary")
        print(f"{'='*100}")

        # 1️⃣ Config summary
        quant_cfg = getattr(model.config, "quantization_config", None)
        if quant_cfg:
            print(f"Quantization config detected: {quant_cfg}")
        else:
            print("No quantization config detected (likely full precision).")

        # 2️⃣ Global stats
        num_linear4bit = sum("Linear4bit" in str(type(m)) for m in model.modules())
        num_linear8bit = sum("Linear8bitLt" in str(type(m)) for m in model.modules())
        num_linear = sum("Linear" in str(type(m)) for m in model.modules())

        print(f"\nTotal Linear layers: {num_linear}")
        print(f" → 4-bit quantized layers: {num_linear4bit}")
        print(f" → 8-bit quantized layers: {num_linear8bit}")

        # 3️⃣ Section-wise quantization breakdown
        print("\nSection-wise Quantization Overview:")
        key_parts = [
            "vision",
            "encoder",
            "decoder",
            "project",
            "language",
            "lm_head",
            "clip",
            "text",
            "embed",
            "embedding",
            "token_emb",
            "word_emb",
        ]
        section_stats = {p: {"4bit": 0, "8bit": 0, "total": 0} for p in key_parts}

        for name, module in model.named_modules():
            if "Linear" in module.__class__.__name__:
                for part in key_parts:
                    if part in name.lower():
                        section_stats[part]["total"] += 1
                        if "Linear4bit" in module.__class__.__name__:
                            section_stats[part]["4bit"] += 1
                        elif "Linear8bitLt" in module.__class__.__name__:
                            section_stats[part]["8bit"] += 1

        for part, stats in section_stats.items():
            if stats["total"] > 0:
                q_type = (
                    "4-bit"
                    if stats["4bit"] > 0
                    else ("8-bit" if stats["8bit"] > 0 else "FP16/32")
                )
                print(
                    f"  • {part:<12} → total: {stats['total']:<3} | "
                    f"4-bit: {stats['4bit']:<3} | 8-bit: {stats['8bit']:<3} | Mode: {q_type}"
                )

        # 4️⃣ Global dtype & device
        try:
            sample_param = next(model.parameters())
            print(f"\nModel dtype:  {sample_param.dtype}")
            print(f"Model device: {sample_param.device}")
        except StopIteration:
            print("No parameters found.")

        # 5️⃣ GPU memory snapshot
        if torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated() / 1e9
            mem_reserved = torch.cuda.memory_reserved() / 1e9
            print(
                f"GPU memory allocated: {mem_alloc:.2f} GB  |  reserved: {mem_reserved:.2f} GB"
            )

        print(f"{'='*100}\n")
