from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
import torch
from datasets import load_dataset
import os
import subprocess

print("Libraries Loaded")
print("CUDA Available: ", torch.cuda.is_available())
print("CUDA Device count: ", torch.cuda.device_count())
print("World size:", os.environ.get("WORLD_SIZE"))
print("Rank:", os.environ.get("RANK"))
print("Local rank:", os.environ.get("LOCAL_RANK"))

for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: ", torch.cuda.get_device_name(i))


class ModelManager:

    def __init__(
        self,
        model_name: str,
        fine_tuned_path: str = "",
        quantize_4bit=False,
        quantize_8bit=False,
    ):

        if model_name == "qwen":
            from Library.local_models.qwen import Qwen

            self.local_model = Qwen(
                fine_tuned_path=fine_tuned_path,
                quantize_4bit=quantize_4bit,
                quantize_8bit=quantize_8bit,
            )

        elif model_name == "phi":
            from Library.local_models.phi import Phi

            self.local_model = Phi(
                fine_tuned_path=fine_tuned_path,
                quantize_4bit=quantize_4bit,
                quantize_8bit=quantize_8bit,
            )

    def freeze_LLM_part(self):
        # Freezing the LLM part
        for name, param in self.model.named_parameters():
            if "model.layers" in name or "lm_head" in name:
                param.requires_grad = False

        trainable = 0
        frozen = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable += param.numel()
            else:
                frozen += param.numel()

        print(f"Trainable params: {trainable:,}")
        print(f"Frozen params:   {frozen:,}")

    def run_inference(self, *args, **kwargs):
        return self.local_model.run_inference(*args, **kwargs)

    def count_inference_token(self, *args, **kwargs):
        return self.local_model.count_inference_token(*args, **kwargs)

    # Outputs in base_model-dataset_name-uuid
    def fine_tune(
        self,
        dataset_name: str,
        preprocess,
        save_path,
        args,
        epoch_num=2,
    ):
        # Loading the dataset
        print("Model name: ", self.local_model.base_model)
        print("Number of Epochs: ", epoch_num)

        processed_dataset = self.local_model.process_dataset(
            dataset_name=dataset_name, preprocess=preprocess, args=args
        )

        self.local_model.tokenizer.padding_side = "left"
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.local_model.tokenizer,
            model=self.local_model.model,
            padding="longest",
        )

        training_args = TrainingArguments(
            output_dir=save_path,
            learning_rate=1e-5,
            num_train_epochs=epoch_num,
            # warmup_ratio=0,
            bf16=True,
            save_strategy="epoch",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            max_grad_norm=1.0,
            logging_steps=2,
            report_to="tensorboard",
        )

        trainer = Trainer(
            model=self.local_model.model,
            args=training_args,
            train_dataset=processed_dataset,
            data_collator=data_collator,
        )

        trainer.train()
        print("Training is finished, merging the outputs from GPU's")

    def _merge_finetune_outputs(self, output_dir):

        checkpoint_root = os.path.abspath(output_dir)

        for ckpt in sorted(os.listdir(checkpoint_root)):
            ckpt_path = os.path.join(checkpoint_root, ckpt)

            print("Merging: ", ckpt_path)
            subprocess.run(
                [
                    "python",
                    os.path.join(ckpt_path, "zero_to_fp32.py"),
                    ckpt_path,
                    os.path.join(ckpt_path, "pytorch_model.bin"),
                ]
            )

            subprocess.run(
                f"mv {ckpt_path}/pytorch_model.bin/* {ckpt_path}/", shell=True
            )

    def load_dataset(self, dataset_name="AI4Math/MathVista", split="testmini"):
        self.dataset_name = dataset_name
        print(self.dataset_name)
        dataset = load_dataset(dataset_name, split=split)
        self.dataset = dataset

    def benchmark(self, get_inputs, compare_outputs, dataset_name="AI4Math/MathVista"):

        self.load_dataset(dataset_name)

        correct = 0
        total = 0
        for row in self.dataset:
            total += 1
            # Getting the image and the prompt from the dataset
            images, prompt = get_inputs(row)
            pred = self.run_inference(images=images, prompt=prompt).strip()
            correct += compare_outputs(row, pred)
            print("Accuracy: ", correct / total)

        return correct / total

        import torch

    def prepare_for_inference(self):
        self.local_model._prepare_for_inference()

    # GPT generated function
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
