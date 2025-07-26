import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import functional as F
import time
import numpy as np
from tqdm import tqdm
import os
import argparse
import datetime
import wandb
import random
import string
import json
from contextlib import nullcontext
from bytelatent.model.blt import ByteLatentTransformerArgs, ByteLatentTransformer
from bytelatent.tokenizers.constants import BOE_ID, BOS_ID, EOS_ID, PAD_ID
from dataloader import DataLoaderBLT


def generate_run_name():
    """Generate a random run name for wandb"""
    adjectives = ["quick", "lazy", "sleepy", "noisy", "hungry", "brave", "bright", "calm", "clever", "fierce"]
    animals = ["fox", "dog", "cat", "wolf", "tiger", "eagle", "shark", "dolphin", "panda", "koala"]
    return f"{random.choice(adjectives)}-{random.choice(animals)}-{random.randint(1, 999)}"


def get_lr(iteration, max_iters, warmup_iters, learning_rate, min_lr, decay_lr=True):
    """
    Learning rate scheduler function
    Args:
        iteration: Current iteration
        max_iters: Total number of iterations
        warmup_iters: Number of warmup iterations
        learning_rate: Peak learning rate
        min_lr: Minimum learning rate
        decay_lr: Whether to decay learning rate after warmup
    Returns:
        lr: Learning rate for current iteration
    """
    # Linear warmup
    if iteration < warmup_iters:
        return learning_rate * (iteration / warmup_iters)
    
    # Decay phase (if enabled)
    if decay_lr:
        # Cosine decay from learning_rate to min_lr
        decay_ratio = (iteration - warmup_iters) / (max_iters - warmup_iters)
        coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))  # Cosine decay
        return min_lr + coeff * (learning_rate - min_lr)
    else:
        # Constant learning rate after warmup
        return learning_rate


def setup(rank, world_size):
    """Setup distributed training environment with proper error handling"""
    try:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ['NCCL_DEBUG'] = 'INFO'  # Add NCCL debugging
        
        # Initialize the process group with timeout
        dist.init_process_group(
            "nccl", 
            rank=rank, 
            world_size=world_size,
            timeout=datetime.timedelta(minutes=2)
        )
        print(f"Rank {rank}: Process group initialized successfully")
    except Exception as e:
        print(f"Rank {rank}: Process group initialization failed: {str(e)}")
        raise


def cleanup():
    """Cleanup distributed training environment"""
    try:
        dist.destroy_process_group()
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")


def unwrap_model(model):
    """Unwrap model from DDP wrapper if needed"""
    return model.module if isinstance(model, DDP) else model


def save_checkpoint(model, optimizer, epoch, global_step, loss, args, is_best=True, filename=None):
    """
    Save model checkpoint with improved error handling
    Args:
        model: The model to save
        optimizer: The optimizer to save
        epoch: Current epoch
        global_step: Global training step
        loss: Current loss value
        args: Training arguments
        is_best: Whether this is the best model so far
        filename: Custom filename (optional)
    """
    try:
        checkpoint_dir = args.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # For DDP models, we need to save the actual model, not the DDP wrapper
        model_state_dict = unwrap_model(model).state_dict()
        
        checkpoint = {
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'global_step': global_step,
            'loss': loss,
            'args': vars(args)  # Convert to dict for better serialization
        }
        
        if filename is None:
            filename = 'model_best.pth' if is_best else f'model_epoch_{epoch+1}.pth'
        
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        torch.save(checkpoint, checkpoint_path)
        
        # Log model checkpoint to wandb if on rank 0
        if dist.get_rank() == 0:
            wandb.save(checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
    except Exception as e:
        print(f"Error saving checkpoint: {str(e)}")


@torch.no_grad()
def validate(model, val_loader, device, patch_lengths, num_steps, ctx=nullcontext()):
    """
    Validate model with mixed precision support
    Args:
        model: The model to validate
        val_loader: Validation data loader
        device: Device to run validation on
        patch_lengths: Patch lengths tensor
        num_steps: Number of validation steps
        ctx: Precision context
    Returns:
        avg_loss: Average validation loss
    """
    model.eval()
    total_loss = 0
    
    try:
        for step in range(num_steps):
            # Get batch
            x, _, _, y, _, _ = val_loader.next_batch_random()
            x = x.to(device)
            y = y.to(device)
            
            # Forward pass with mixed precision
            with ctx:
                logits = model(x, patch_lengths)
                # Calculate loss
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=PAD_ID)
            
            # All-reduce loss across all processes
            if dist.is_initialized():
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                loss = loss / dist.get_world_size()
                
            total_loss += loss.item()
    except Exception as e:
        print(f"Error during validation: {str(e)}")
        return float('inf')  # Return a high loss value to avoid saving bad checkpoints
    
    # Cleanup memory
    if device != 'cpu':
        torch.cuda.empty_cache()
        
    return total_loss / num_steps


def train(rank, world_size, args):
    # Setup distributed training
    print(f"Rank {rank}: Starting setup")
    setup(rank, world_size)
    print(f"Rank {rank}: Setup complete")
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"Rank {rank}: Random seed set")
    
    # Flag for master process
    master_process = rank == 0
    
    # Initialize wandb only on rank 0
    if master_process:
        run_name = generate_run_name()
        wandb.init(
            project="blt final",
            name=run_name,
            config=vars(args)
        )
        print(f"Rank {rank}: Initialized wandb with run name: {run_name}")
    
    # Precision settings - similar to first code
    torch.set_float32_matmul_precision('high')
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if not torch.cuda.is_available() else torch.amp.autocast(device_type='cuda', dtype=ptdtype)
    scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))
    
    if master_process:
        print(f"Using precision: {dtype}")
    
    # Create model
    print(f"Rank {rank}: Creating model")
    model_args = ByteLatentTransformerArgs(
        seed=args.seed,
        vocab_size=4096,
        dim=256,
        n_layers_global=8,
        n_layers_local_decoder=4,
        n_layers_local_encoder=4,
        patch_size=8,
        patching_mode="static",
        tie_local_encoder_decoder_logits=False,
        patch_in_forward=False,
        max_encoder_seq_length=args.seq_length,
        pad_to_max_length=True,
        patching_threshold=3.1439168453216553,
        encoder_hash_byte_group_size=[4],
        encoder_hash_byte_group_vocab=50002,
        encoder_hash_byte_group_nb_functions=3,
        encoder_enable_byte_ngrams=False,
        cross_attn_encoder=True,
        cross_attn_decoder=True,
        cross_attn_window_encoder=None,
        cross_attn_window_decoder=None,
        dim_local_encoder=128,
        dim_local_decoder=128,
        cross_attn_k=2,
        cross_attn_nheads=4,
        cross_attn_all_layers_decoder=True,
        cross_attn_all_layers_encoder=True,
        cross_attn_use_flex_attention=False,
        cross_attn_init_by_pooling=True,
        log_patch_lengths=True,
        non_linearity="swiglu",
        use_rope=True,
        recompute_fc1_out=False,
        recompute_fc3_out=False,
        recompute_attn=False,
        custom_bwd=False,
        layer_ckpt="none",
        use_local_encoder_transformer=True,
        init_use_gaussian=True,
        init_use_depth="current",
        attn_impl="sdpa",
        attn_bias_type="causal",
        alpha_depth="disabled",
        max_length=args.seq_length,
        local_attention_window_len=args.seq_length,
        max_seqlen=args.seq_length,
        downsampling_by_pooling="max",
    )
    
    model = ByteLatentTransformer(model_args)
    print(f"Rank {rank}: Model created")
    
    # Move model to current GPU
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Rank {rank}: Model moved to device {device}")
    
    # Compile the model if PyTorch 2.0+ is available and enabled
    if args.compile and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            if master_process:
                print("Model successfully compiled with torch.compile")
        except Exception as e:
            if master_process:
                print(f"Warning: Failed to compile model: {e}. Continuing without compilation.")
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)
    print(f"Rank {rank}: Model wrapped with DDP")
    
    # Print model size on master process
    if master_process:
        num_params = sum(p.numel() for p in unwrap_model(model).parameters())
        print(f"Model size: {num_params / 1e6:.2f} M parameters")
    
    # Create data loaders
    print(f"Rank {rank}: Creating data loaders")
    try:
        train_loader = DataLoaderBLT(
            batch_size=args.batch_size, 
            seq_length=args.seq_length, 
            split="train", 
            process_rank=rank, 
            num_processes=world_size
        )
        val_loader = DataLoaderBLT(
            batch_size=args.batch_size, 
            seq_length=args.seq_length, 
            split="val", 
            process_rank=rank, 
            num_processes=world_size
        )
        print(f"Rank {rank}: Data loaders created")
    except Exception as e:
        print(f"Error creating data loaders: {str(e)}")
        cleanup()
        raise
    
    # Calculate effective batch size
    global_batch_size = args.batch_size * args.gradient_accumulation_steps * world_size
    if master_process:
        print(f"Global batch size: {global_batch_size} (local_bs={args.batch_size}, "
              f"grad_accum={args.gradient_accumulation_steps}, world_size={world_size})")
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)  # Use better beta values from first code
    )
    
    # Create patch lengths tensor - this should be moved to the prepare_batch function ideally
    patch_lengths = torch.full((args.batch_size, 64), 8).to(device)
    patch_lengths[:, 0] = 1
    patch_lengths[:, -1] = 10
    patch_lengths[:, 2] = 11
    patch_lengths[:, 1] = 10
    
    # Calculate training hyperparameters
    total_steps = args.epochs * args.steps_per_epoch
    warmup_steps = args.warmup_steps if args.warmup_steps > 0 else int(0.1 * total_steps)
    min_lr = args.learning_rate * args.min_lr_factor
    
    # Create checkpoint directory and save config
    if master_process:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        config = {
            'batch_size': args.batch_size,
            'gradient_accumulation_steps': args.gradient_accumulation_steps,
            'effective_batch_size': global_batch_size,
            'learning_rate': args.learning_rate,
            'total_steps': total_steps,
            'warmup_steps': warmup_steps,
            'min_lr_factor': args.min_lr_factor,
            'decay_lr': args.decay_lr,
            'grad_clip': args.clip_grad,
            'dtype': dtype,
            'model_size': num_params
        }
        with open(os.path.join(args.checkpoint_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    
    if args.resume_from and os.path.exists(args.resume_from):
        try:
            checkpoint = torch.load(args.resume_from, map_location=device)
            unwrap_model(model).load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            global_step = checkpoint.get('global_step', 0)
            best_val_loss = checkpoint.get('loss', float('inf'))
            if master_process:
                print(f"Resuming from checkpoint: {args.resume_from}")
                print(f"Starting from epoch {start_epoch}, global step {global_step}")
                print(f"Best validation loss so far: {best_val_loss:.4f}")
        except Exception as e:
            if master_process:
                print(f"Error loading checkpoint: {str(e)}. Starting from scratch.")
    
    # Create a CSV log file for training progress
    if master_process:
        log_file = os.path.join(args.checkpoint_dir, 'training_log.csv')
        with open(log_file, 'w') as f:
            f.write("epoch,iteration,train_loss,val_loss,learning_rate,time_ms\n")
    
    # Training loop
    start_time = time.time()
    patience_counter = 0
    t0 = time.time()  # For timing
    
    print(f"Rank {rank}: Starting training loop")
    
    # Pre-fetch first batch outside the loop
    try:
        batch = train_loader.next_batch_random()
    except Exception as e:
        print(f"Error pre-fetching first batch: {e}")
        cleanup()
        raise
    
    try:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            epoch_loss = 0
            num_batches = args.steps_per_epoch
            
            # Only use tqdm on rank 0 to avoid duplicate progress bars
            if master_process:
                pbar = tqdm(total=num_batches, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch")
            
            for step in range(num_batches):
                # Get learning rate for this step
                iteration = epoch * num_batches + step
                lr = get_lr(
                    iteration, 
                    total_steps, 
                    warmup_steps, 
                    args.learning_rate, 
                    min_lr, 
                    args.decay_lr
                )
                
                # Update learning rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                
                # Accumulated loss for logging
                total_loss = 0
                
                # Reset gradients
                optimizer.zero_grad(set_to_none=True)
                
                # Gradient accumulation loop
                for micro_step in range(args.gradient_accumulation_steps):
                    try:
                        # Setup gradient synchronization for DDP
                        if isinstance(model, DDP):
                            # Only sync gradients at the last micro step
                            model.require_backward_grad_sync = (
                                micro_step == args.gradient_accumulation_steps - 1
                            )
                        
                        # Get batch (already pre-fetched for first iteration)
                        if not (epoch == start_epoch and step == 0 and micro_step == 0):
                            batch = train_loader.next_batch_random()
                        
                        # Unpack and move to device
                        x, _, _, y, _, _ = batch
                        x = x.to(device)
                        y = y.to(device)
                        
                        # Forward pass with mixed precision
                        with ctx:
                            logits = model(x, patch_lengths)
                            # Calculate loss
                            loss = F.cross_entropy(
                                logits.reshape(-1, logits.size(-1)), 
                                y.reshape(-1), 
                                ignore_index=PAD_ID
                            )
                            # Scale loss for gradient accumulation
                            loss = loss / args.gradient_accumulation_steps
                        
                        # Backward pass with gradient scaling for float16
                        scaler.scale(loss).backward()
                        
                        # Track loss for logging
                        total_loss += loss.item() * args.gradient_accumulation_steps
                        
                    except Exception as e:
                        print(f"Rank {rank}: Error in micro step {micro_step}: {str(e)}")
                        raise
                
                # Gradient clipping
                if args.clip_grad > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                
                # Update weights with scaler
                scaler.step(optimizer)
                scaler.update()
                
                # Track metrics
                epoch_loss += total_loss
                global_step += 1
                
                # Update progress bar and log metrics only on rank 0
                if master_process:
                    # Timing information
                    if step % args.log_interval == 0:
                        if device != 'cpu':
                            torch.cuda.synchronize()
                        t1 = time.time()
                        dt = t1 - t0
                        ms_per_iter = dt * 1000 / max(1, args.log_interval)
                        t0 = t1
                        
                        # Log to CSV
                        with open(log_file, 'a') as f:
                            f.write(f"{epoch+1},{global_step},{total_loss:.6f},nan,{lr:.6f},{ms_per_iter:.2f}\n")
                    
                    # Update tqdm progress bar
                    pbar.set_postfix(loss=total_loss, lr=lr)
                    pbar.update(1)
                    
                    # Log to wandb
                    wandb.log({
                        "train/loss": total_loss,
                        "train/learning_rate": lr,
                        "train/epoch": epoch + (step + 1) / num_batches,
                        "train/global_step": global_step,
                        "train/ms_per_iter": ms_per_iter if step % args.log_interval == 0 else None
                    })
            
            # Close progress bar if on rank 0
            if master_process:
                pbar.close()
                
                # End of epoch
                avg_epoch_loss = epoch_loss / num_batches
                print(f"Epoch {epoch+1} completed | Avg loss: {avg_epoch_loss:.4f}")
                
                # Log epoch metrics to wandb
                wandb.log({
                    "train/epoch_loss": avg_epoch_loss,
                    "train/epoch": epoch + 1
                })
            
            # Evaluation
            if (epoch + 1) % args.eval_every == 0:
                # DDP: ensure all processes are synchronized before evaluation
                if dist.is_initialized():
                    dist.barrier()
                
                # Run validation
                val_loss = validate(model, val_loader, device, patch_lengths, args.val_steps, ctx)
                
                # Log validation results
                if master_process:
                    print(f"Epoch {epoch+1} | Validation loss: {val_loss:.4f}")
                    
                    # Log validation metrics to wandb
                    wandb.log({
                        "val/loss": val_loss,
                        "val/epoch": epoch + 1
                    })
                    
                    # Log to CSV
                    with open(log_file, 'a') as f:
                        f.write(f"{epoch+1},{global_step},{avg_epoch_loss:.6f},{val_loss:.6f},{lr:.6f},0\n")
                    
                    # Save checkpoint if it's the best model so far
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        save_checkpoint(model, optimizer, epoch, global_step, val_loss, args, is_best=True)
                        
                        # Log best model metrics to wandb
                        wandb.log({
                            "val/best_loss": val_loss,
                            "val/best_epoch": epoch + 1
                        })
                        
                        print(f"New best model saved! Validation loss: {val_loss:.4f}")
                    else:
                        patience_counter += 1
                        if args.patience > 0 and patience_counter >= args.patience:
                            print(f"Early stopping triggered after {args.patience} evaluations without improvement")
                            break
            
            # Save regular checkpoint - only on master process
            if master_process and (epoch + 1) % args.save_every == 0:
                save_checkpoint(
                    model, 
                    optimizer, 
                    epoch, 
                    global_step, 
                    avg_epoch_loss, 
                    args, 
                    is_best=False, 
                    filename=f"checkpoint_epoch_{epoch+1}.pt"
                )
        
        # Save final model
        if master_process:
            save_checkpoint(
                model, 
                optimizer, 
                args.epochs-1, 
                global_step, 
                best_val_loss, 
                args, 
                is_best=False, 
                filename="final_checkpoint.pt"
            )
            
            print("Training completed!")
            total_time = time.time() - start_time
            print(f"Total training time: {total_time:.2f} seconds")
            
            # Log final metrics and finish wandb run
            wandb.log({
                "train/total_time": total_time,
                "train/best_val_loss": best_val_loss
            })
            wandb.finish()
    
    except Exception as e:
        print(f"Error during training: {str(e)}")
        
        # Try to save checkpoint on error (only on rank 0)
        if master_process:
            try:
                save_checkpoint(
                    model, 
                    optimizer, 
                    epoch, 
                    global_step, 
                    float('inf'), 
                    args, 
                    is_best=False, 
                    filename="error_checkpoint.pt"
                )
            except Exception as save_error:
                print(f"Failed to save error checkpoint: {str(save_error)}")
    
    finally:
        # Clean up DDP - ensure this happens even on exceptions
        if dist.is_initialized():
            try:
                dist.barrier()  # Synchronize before destroying process group
                cleanup()
            except Exception as e:
                print(f"Error during DDP cleanup: {str(e)}")
        
        # Final memory cleanup
        if device != 'cpu':
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description='Train ByteLatentTransformer with improved implementation')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size per GPU')
    parser.add_argument('--seq_length', type=int, default=512, help='Sequence length')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Peak learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--steps_per_epoch', type=int, default=1000, help='Steps per epoch')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2, 
                        help='Number of batches to accumulate gradients')
    parser.add_argument('--clip_grad', type=float, default=1.0, help='Gradient clipping (0 to disable)')
    parser.add_argument('--seed', type=int, default=777, help='Random seed')
    parser.add_argument('--warmup_steps', type=int, default=0, 
                        help='Warmup steps (0 for 10% of total steps)')
    parser.add_argument('--min_lr_factor', type=float, default=0.1, 
                        help='Minimum learning rate as fraction of peak learning rate')
    parser.add_argument('--decay_lr', type=bool, default=True, 
                        help='Whether to use learning rate decay after warmup')
    parser.add_argument('--compile', type=bool, default=False, 
                        help='Use torch.compile for faster training (requires PyTorch 2.0+)')
    
    # Logging and checkpoint parameters
    parser.add_argument('--log_interval', type=int, default=10, help='Logging frequency (in steps)')
    parser.add_argument('--eval_every', type=int, default=1, help='Evaluate every N epochs')
    parser.add_argument('--val_steps', type=int, default=1000, help='Number of validation steps')
    parser.add_argument('--save_every', type=int, default=1, help='Save checkpoint every N epochs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--patience', type=int, default=10, 
                        help='Early stopping patience (0 to disable)')
    parser.add_argument('--resume_from', type=str, default=None, 
                        help='Path to checkpoint to resume training from')
    
    # Distributed training parameters
    parser.add_argument('--num_gpus', type=int, default=torch.cuda.device_count() if torch.cuda.is_available() else 1, 
                        help='Number of GPUs to use')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.batch_size <= 0:
        parser.error("batch_size must be positive")
    if args.gradient_accumulation_steps <= 0:
        parser.error("gradient_accumulation_steps must be positive")
    if args.learning_rate <= 0:
        parser.error("learning_rate must be positive")
    
    # Set default num_gpus to available GPUs if not specified
    world_size = min(args.num_gpus, torch.cuda.device_count()) if torch.cuda.is_available() else 1
    
    # For single GPU or CPU, don't use distributed training
    if world_size == 1:
        train(0, 1, args)
    else:
        # Launch processes for distributed training
        mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == '__main__':
    main()