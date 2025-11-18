from Library.model_manager import ModelManager
from Library.dataset_process import AdapterProvider
from Library.prompts import context_maker
import argparse
import random
import time
import numpy as np
import sys

context_maker = (
    lambda example: f"""
You are a helpful and patient math tutor assisting a student.
help them understand how to solve the problem and ask them questions accordingly, 
and avoid simply giving away the final answer immediately.

Below is contextual information for the current problem:

Problem Statement:
{example['question']}

Reference Solution (for your understanding, not to copy directly):
{example['answer']}

When you respond, reason carefully and clearly.
Focus on explaining the reasoning steps at the student's level of understanding.
"""
    # If the question involves visual input, describe and reason about the image explicitly.
    # End your response with the final numerical or symbolic answer when appropriate.
)

if __name__ == "__main__":

    # This is to make datasetnames easier to pass as arguments
    dataset_dict = {
        "mathvista": "AI4Math/MathVista",
        "mathdial": "eth-nlped/mathdial",
        "mathvision": "MathLLMs/MathVision",
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=["train", "benchmark", "inference", "token-count"]
    )
    parser.add_argument(
        "--train-dataset",
        type=str,
        default="mathvision",
        help="Supported datasets are: mathvista, mathdial",
    )
    parser.add_argument(
        "--benchmark-dataset",
        type=str,
        default="mathvista",
        help="Supported datasets are: mathvista, mathdial",
    )
    parser.add_argument(
        "--from-finetuned",
        type=str,
        default=None,
        help="The path to the checkpoint of the fine-tuned model, ex:./Models/phi-3.5_MathVista/checkpoint-1000",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./Models",
        help="The output directory for fine-tuned models",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        help="Name your run(this will be used in the saved place s.t ./Model/{run_name})",
        default="TESTING",
    )
    parser.add_argument(
        "--process-dataset",
        default=False,
        type=bool,
        help="If you need to process the dataset or if it's ready for training",
    )
    parser.add_argument("--quantize-4bit", default=False, type=bool)
    parser.add_argument("--quantize-8bit", default=False, type=bool)
    parser.add_argument("--epoch-num", type=int, default=2)
    parser.add_argument("--dataset-split", type=str, default="testmini")
    parser.add_argument("--model-name", type=str, default="phi")
    parser.add_argument("--direct-to", type=str)

    args = parser.parse_args()

    if args.direct_to:
        log_file = open(args.direct_to + ".txt", "w")
        sys.stdout = log_file

    if args.run_name:
        full_fine_tune_path = args.output_dir + "/" + args.run_name
    else:
        full_fine_tune_path = args.output_dir

    if args.mode == "train" and (not args.train_dataset or not args.run_name):
        parser.error("--train-dataset is required when mode is 'train'")

    if args.mode == "benchmark" and not args.benchmark_dataset:
        parser.error("--benchmark-dataset is required when mode is 'benchmark'")

    if args.from_finetuned:
        print("Loading model from: ", args.from_finetuned)
        manager: ModelManager = ModelManager(
            fine_tuned_path=args.from_finetuned,
            model_name=args.model_name,
            quantize_4bit=args.quantize_4bit,
            quantize_8bit=args.quantize_8bit,
        )
    else:
        manager: ModelManager = ModelManager(
            model_name=args.model_name,
            quantize_4bit=args.quantize_4bit,
            quantize_8bit=args.quantize_8bit,
        )

    if args.mode == "train":
        print("Preparing the training!")
        dataset_name = args.train_dataset
        manager.fine_tune(
            dataset_dict[dataset_name],
            preprocess=args.process_dataset,
            save_path=full_fine_tune_path,
            epoch_num=args.epoch_num,
            args=args,
        )

    elif args.mode == "benchmark":
        print("Preparing Benchmark!")
        dataset_name = args.benchmark_dataset
        provider = AdapterProvider(dataset_name=dataset_name)
        manager.benchmark(
            provider.get_inputs,
            provider.compare_outputs,
            dataset_name=dataset_dict[dataset_name],
        )

    elif args.mode == "inference":
        print("Setting up the inference mode, 'n' for next question")
        manager.load_dataset(dataset_name=dataset_dict["mathvision"], split="test")
        # manager.load_dataset(dataset_name=dataset_dict["mathvision"], split="testmini")
        # before = time.time()
        manager.prepare_for_inference()
        # ex_idx = random.randint(0, len(manager.dataset) - 1)
        ex_idx = 167
        example = manager.dataset[ex_idx]
        image = example["decoded_image"]
        context = context_maker(example)
        print(example["id"])

        # system = {
        #     "role": "system",
        #     "content": [
        #         {"type": "text", "text": context},
        #         {"type": "image", "image": image},
        #     ],
        # }
        system = {"role": "assistant", "content": context}

        user = input(f"Type the answer to this question: {example['question']}\n")
        # user = "I think the answer is A"
        messages = [system]

        messages.append({"role": "user", "content": user})
        response = manager.run_inference(messages=messages)
        print("Model: ", response + "\n")
        messages.append({"role": "assistant", "content": response})
        while user != "n":
            user = input()
            messages.append({"role": "user", "content": user})
            response = manager.run_inference(messages=messages)
            print("Model: ", response + "\n")
            messages.append({"role": "assistant", "content": response})

    elif args.mode == "token-count":

        manager.load_dataset(dataset_name=dataset_dict["mathvista"], split="testmini")
        manager.prepare_for_inference()

        total_tokens = []
        print("Total to process: ", len(manager.dataset))
        for idx in range(len(manager.dataset)):

            example = manager.dataset[idx]
            image = example["decoded_image"]
            context = context_maker(example)

            system = {"role": "assistant", "content": context}
            messages = [system]
            user_response = "I'm not sure how to begin the question"

            messages.append({"role": "user", "content": user_response})

            response, count = manager.count_inference_token(messages=messages)
            total_tokens.append(count)
            print(f"Response: {idx} ", response[:50], "...")
            print("Count: ", count)
            sys.stdout.flush()

        tokens = np.array(total_tokens)
        # GPT print
        print("=== Token Statistics ===")
        print(f"Mean:       {tokens.mean():.2f}")
        print(f"Median:     {np.median(tokens):.2f}")
        print(f"Std Dev:    {tokens.std():.2f}")
        print(f"Min:        {tokens.min()}")
        print(f"Max:        {tokens.max()}")
        print("Whole array: ", tokens)
