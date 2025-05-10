import os
import shutil
import tarfile
import argparse

from pathlib import Path
from preprocess import preprocess
from trainer_pti import main

# from predict import SDXL_MODEL_CACHE, SDXL_URL, download_weights
SDXL_MODEL_CACHE = "stabilityai/stable-diffusion-xl-base-1.0"  # Bạn có thể sửa lại cho phù hợp

OUTPUT_DIR = "training_out"

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_model_name_or_path", type=str, default=SDXL_MODEL_CACHE, help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--input_images", type=str, required=True, help="Path to the input images directory or zip file.")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Path to save the output model.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=768, help="Resolution of the images.")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--num_train_epochs", type=int, default=4000, help="Number of training epochs.")
    parser.add_argument("--max_train_steps", type=int, default=1000, help="Maximum number of training steps.")
    parser.add_argument("--is_lora", action="store_true", default=False, help = "Use LoRA for training.")
    parser.add_argument("--unet_learning_rate", type=float, default=1e-6, help = "Learning rate for the UNet.")
    parser.add_argument("--ti_lr", type=float, default=3e-4, help = "Learning rate for the text encoder with textual inversion.")
    parser.add_argument("--lora_lr", type=float, default=1e-4, help = "Scaling of learning rate for training LoRA embeddings. Don't alter unless you know what you're doing.")
    parser.add_argument("--lora_rank", type=int, default=32, help = "Rank of the LoRA layers. Don't alter unless you know what you're doing.")
    parser.add_argument("--lr_scheduler", type=str, choices=["constant", "linear"], default="constant", help = "Learning rate scheduler to use.")
    parser.add_argument("--lr_warmup_steps", type=int, default=100, help = "Number of warmup steps for the learning rate scheduler.")
    parser.add_argument("--token_string", type=str, default="TOK", help = "A unique string that will be trained to refer to the concept in the input images.")
    parser.add_argument("--caption_prefix", type=str, default="a photo of TOK, ", help = "Prefix for the captions when use automatically generating captions, must include the token string in the caption")
    parser.add_argument("--mask_target_prompts", type=str, default=None, help = "Prompt that describes part of the image that you will find important.")
    parser.add_argument("--crop_based_on_salience", action="store_true", help = "Whether to crop the images based on salience.")
    parser.add_argument("--use_face_detection_instead", action="store_true", help = "Whether to use face detection instead of using CLIPSeg")
    parser.add_argument("--clipseg_temperature", type=float, default=1.0, help = "How blurry you want the CLIPSeg mask to be")
    parser.add_argument("--verbose", action="store_true", help = "verbose output")
    parser.add_argument("--checkpointing_steps", type=int, default=999999, help = "Checkpointing steps, set to a large number to disable checkpointing")
    parser.add_argument("--input_images_filetype", type=str, default="infer", choices=["zip", "tar", "infer"])

    return parser.parse_args()

def main_train(args):
    token_map = args.token_string + ":2"
    inserting_list_tokens = token_map.split(",")
    token_dict = {}
    all_token_lists = []
    running_tok_cnt = 0
    for token in inserting_list_tokens:
        n_tok = int(token.split(":")[1])
        token_dict[token.split(":")[0]] = "".join([f"<s{i + running_tok_cnt}>" for i in range(n_tok)])
        all_token_lists.extend([f"<s{i + running_tok_cnt}>" for i in range(n_tok)])
        running_tok_cnt += n_tok

    input_dir = preprocess(
        input_images_filetype=args.input_images_filetype,
        input_zip_path=Path(args.input_images),
        caption_text=args.caption_prefix,
        mask_target_prompts=args.mask_target_prompts,
        target_size=args.resolution,
        crop_based_on_salience=args.crop_based_on_salience,
        use_face_detection_instead=args.use_face_detection_instead,
        temp=args.clipseg_temperature,
        substitution_tokens=list(token_dict.keys()),
    )

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)

    main(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        instance_data_dir=os.path.join(input_dir, "captions.csv"),
        output_dir = args.output_dir,
        seed=args.seed,
        resolution=args.resolution,
        train_batch_size=args.train_batch_size,
        num_train_epochs=args.num_train_epochs,
        max_train_steps=args.max_train_steps,
        gradient_accumulation_steps=1,
        unet_learning_rate=args.unet_learning_rate,
        ti_lr=args.ti_lr,
        lora_lr=args.lora_lr,
        lr_scheduler=args.lr_scheduler,
        lr_warmup_steps=args.lr_warmup_steps,
        token_dict=token_dict,
        inserting_list_tokens=all_token_lists,
        verbose=args.verbose,
        checkpointing_steps=args.checkpointing_steps,
        scale_lr=False,
        max_grad_norm=1.0,
        allow_tf32=True,
        mixed_precision="bf16",
        device="cuda:0",
        lora_rank=args.lora_rank,
        is_lora=args.is_lora,
    )

    out_path = "trained_model.tar"
    with tarfile.open(out_path, "w") as tar:
        for file_path in Path(args.output_dir).rglob("*"):
            print(file_path)
            arcname = file_path.relative_to(args.output_dir)
            tar.add(file_path, arcname=arcname)

if __name__ == "__main__":
    args = parse_args()
    main_train(args)
