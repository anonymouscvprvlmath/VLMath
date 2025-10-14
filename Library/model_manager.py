from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
import torch
from datasets import load_dataset
import os
import subprocess
from Library.local_models.phi import Phi
from Library.local_models.qwen import Qwen

print("Libraries Loaded")
print("CUDA Available: ", torch.cuda.is_available())
print("CUDA Device count: ", torch.cuda.device_count())
print("World size:", os.environ.get("WORLD_SIZE"))
print("Rank:", os.environ.get("RANK"))
print("Local rank:", os.environ.get("LOCAL_RANK"))

for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: ", torch.cuda.get_device_name(i))


class ModelManager():

    def __init__(self, fine_tuned_path = ""):
        
        # base_model = "microsoft/phi-3-mini-4k-instruct"
        # self.local_model = Phi(fine_tuned_path=fine_tuned_path)
        self.local_model = Qwen(fine_tuned_path=fine_tuned_path)

    def freeze_LLM_part(self):
        # Freezing the LLM part
        for name, param in self.model.named_parameters():
            if "model.layers" in name or "lm_head" in name:
                param.requires_grad = False

        trainable = 0
        frozen = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable+= param.numel()
            else:
                frozen += param.numel()
                 
        print(f"Trainable params: {trainable:,}")
        print(f"Frozen params:   {frozen:,}")  

    # images is a list of image, prompt is str
    # def run_inference(self, prompt = "", images = [], messages = []):
    #     return self.local_model.run_inference(prompt=prompt, images=images, messages=messages) 

    def run_inference(self, *args, **kwargs):
        return self.local_model.run_inference(*args, **kwargs)

    # Outputs in base_model-dataset_name-uuid
    def fine_tune(self, dataset_name:str, split:str, preprocess, save_path, epoch_num=2):
        # Loading the dataset
        print("Model name: ", self.local_model.base_model)
        print("Number of Epochs: ", epoch_num)

        processed_dataset = self.local_model.process_dataset(dataset_name=dataset_name, split=split, preprocess=preprocess)
        print(processed_dataset[0])

        self.local_model.tokenizer.padding_side = "left"
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.local_model.tokenizer,
            model=self.local_model.model,
            padding="longest"
        ) 
 
        training_args = TrainingArguments(
            output_dir = save_path,
            learning_rate = 1e-5,
            num_train_epochs = epoch_num,
            warmup_steps=50,
            bf16=True,
            save_steps = 1500,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            max_grad_norm = 1.0,
            logging_steps = 20,
            report_to="tensorboard"
        )

        trainer = Trainer(
            model = self.local_model.model,
            args = training_args,
            train_dataset = processed_dataset,
            data_collator = data_collator
        )


        trainer.train()
        print("Training is finished, merging the outputs from GPU's")

    def _merge_finetune_outputs(self, output_dir):

        checkpoint_root = os.path.abspath(output_dir)

        for ckpt in sorted(os.listdir(checkpoint_root)):
            ckpt_path = os.path.join(checkpoint_root, ckpt)

            print("Merging: ", ckpt_path)
            subprocess.run(["python",
                os.path.join(ckpt_path, "zero_to_fp32.py"),
                ckpt_path,
                os.path.join(ckpt_path, "pytorch_model.bin"),])
 
            subprocess.run(f"mv {ckpt_path}/pytorch_model.bin/* {ckpt_path}/", shell=True)

    def load_dataset(self, dataset_name = "AI4Math/MathVista", split="testmini"):
        self.dataset_name = dataset_name
        print(self.dataset_name)
        dataset = load_dataset(dataset_name, split=split)
        self.dataset = dataset

    def benchmark(self, get_inputs, compare_outputs, dataset_name="AI4Math/MathVista"):
        
        self.load_dataset(dataset_name)

        correct = 0
        total = 0
        for row in self.dataset:
            total+=1
            # Getting the image and the prompt from the dataset
            images, prompt = get_inputs(row)
            pred = self.run_inference(images=images, prompt=prompt).strip()
            correct+= compare_outputs(row, pred) 
            print("Accuracy: ", correct / total)

        return correct / total


