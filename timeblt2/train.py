import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # must be BEFORE torch/TF import
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.optim as optim
import torch.nn.functional as F
from bytelatent.model.blt import ByteLatentTransformerArgs, ByteLatentTransformer
from chronos import MeanScaleUniformBins, ChronosConfig
from utils.train_utils import build_tokenizer, create_static_patch_lengths, build_dataloader, validate, get_lr
from TLLM_data_provider.data_factory import data_provider
# from data_provider.data_factory import data_provider
import argparse
from tqdm import tqdm
import time
import json
import numpy as np
from pathlib import Path

torch.cuda.set_device(0)   # 0 here means "the first visible GPU", i.e. physical #3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from bytelatent.tokenizers.constants import PAD_ID
from utils.eval_utils import evaluation

## Training Args
vocab_size = 1024
quant_range = 10
batch_size = 2
seq_len = 512
label_len = 0
pred_len = 96
learning_rate = 5e-4
weight_decay = 1e-2
epochs = 500  # Increased for early stopping
grad_accumulation_steps = 1
clip_grad = 1.0
seed = 42
warmup_steps = 0
min_lr_factor = 0.1
decay_lr = True
compile = True
output_dir = "output"
save_every = 5
eval_every = 10  # Evaluate every 5 epochs
patience = 6   # Early stopping patience
compile = False
dataset_name = 'ETTh1'
features = 'S'

# Create output directory
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Set Data Loaders
train_dataset, train_loader = build_dataloader(dataset_name, features, seq_len, seq_len - 1, 1, 'train', batch_size, pretrain=True)
validate_dataset, validate_loader = build_dataloader(dataset_name, features, seq_len, seq_len - 1, 1, 'val', batch_size, pretrain=True)
print(f"Dataset: {dataset_name}, Features: {features}, Batch Size: {batch_size}, Seq Len: {seq_len}")

## Set model args
model_args = ByteLatentTransformerArgs(
    seed=42,
    vocab_size=vocab_size,                       # Small byte-level vocab
    max_length=seq_len,                        # Max full sequence length
    max_seqlen=seq_len,
    max_encoder_seq_length=seq_len,
    local_attention_window_len=seq_len,        # Local window, 128 is sufficient for small models

    dim_global=32,                        # Lower than default 512
    dim_local_encoder=16,
    dim_local_decoder=16,

    n_layers_global=2,
    n_layers_local_encoder=2,
    n_layers_local_decoder=2,

    n_heads_global=2,                      # Reduce heads
    n_heads_local_encoder=2,
    n_heads_local_decoder=2,

    patch_size=8,
    patch_in_forward=False,                # Patch in forward pass
    patching_batch_size=256,
    patching_device="cuda",               # Use CPU for patching in small model
    patching_mode="entropy",
    patching_threshold=3.0,
    max_patch_length=16,
    monotonicity=True,            # Monotonic patching
    pad_to_max_length=True,

    cross_attn_encoder=True,
    cross_attn_decoder=True,
    cross_attn_k=2,
    cross_attn_nheads=2,
    cross_attn_all_layers_encoder=True,
    cross_attn_all_layers_decoder=True,
    cross_attn_use_flex_attention=False,
    cross_attn_init_by_pooling=True,

    encoder_hash_byte_group_size=[8,9,10],   # Fewer hash sizes
    encoder_hash_byte_group_vocab=2**8,
    encoder_hash_byte_group_nb_functions=1,
    encoder_enable_byte_ngrams=False,

    non_linearity="swiglu",
    use_rope=True,
    attn_impl="sdpa",                      # Efficient PyTorch attention
    attn_bias_type="causal",

    dropout=0.1,
    layer_ckpt="none",                     # No checkpointing in small model
    init_use_gaussian=True,
    init_use_depth="current",
    alpha_depth="disabled",
    log_patch_lengths=True,

    downsampling_by_pooling="max",         # Efficient downsampling
    use_local_encoder_transformer=True,
    share_encoder_decoder_emb=True         # Save memory if possible
)

model = ByteLatentTransformer(model_args)
model = model.to(device)
if compile:
    model = torch.compile(model)

# n of params in model in millions
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters in model: {count_parameters(model) / 1e6:.2f}M")

patch_lengths = create_static_patch_lengths(batch_size=batch_size, seq_len=seq_len)

optimizer = optim.AdamW(
    model.parameters(), 
    lr=5e-4, 
    weight_decay=0.01,
    betas=(0.9, 0.95)  # Use better beta values from first code
)
optimizer.zero_grad(set_to_none=True)

torch.manual_seed(model_args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(model_args.seed)
torch.set_float32_matmul_precision('high')
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))
print(f"Using precision: {dtype}")

class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        """Save model weights"""
        self.best_weights = model.state_dict().copy()

class TrainingLogger:
    """Logger for training metrics"""
    def __init__(self, output_dir, dataset_name):
        self.output_dir = Path(output_dir)
        self.dataset_name = dataset_name
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'eval_results': []
        }
        
    def log_epoch(self, epoch, train_loss, val_loss, lr, eval_results=None):
        self.history['epoch'].append(int(epoch))
        self.history['train_loss'].append(float(train_loss))
        self.history['val_loss'].append(float(val_loss))
        self.history['learning_rate'].append(float(lr))
        if eval_results:
            # Convert eval_results to JSON-serializable format
            serializable_results = self._make_json_serializable(eval_results)
            self.history['eval_results'].append({
                'epoch': int(epoch),
                'results': serializable_results
            })
        
        # Save to file
        with open(self.output_dir / f'training_history_{self.dataset_name}_{features}.json', 'w') as f:
            json.dump(self.history, f, indent=4)
    
    def _make_json_serializable(self, obj):
        """Convert numpy/torch types to JSON serializable Python types"""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):  # PyTorch tensors
            return float(obj.item())
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)  # Fallback to string representation
    
    def print_summary(self):
        """Print training summary"""
        print(f"\n{'='*60}")
        print(f"📊 TRAINING SUMMARY")
        print(f"{'='*60}")
        print(f"Total epochs: {len(self.history['epoch'])}")
        print(f"Best validation loss: {min(self.history['val_loss']):.6f}")
        print(f"Final validation loss: {self.history['val_loss'][-1]:.6f}")
        if self.history['eval_results']:
            print(f"Evaluations performed: {len(self.history['eval_results'])}")

def training_with_early_stopping(model, train_loader, validate_loader, patch_lengths, device, 
                                epochs=50, learning_rate=5e-4, weight_decay=1e-2, 
                                grad_accumulation_steps=1, clip_grad=1.0, warmup_steps=0, 
                                min_lr_factor=0.1, decay_lr=True, output_dir='output',
                                dataset_name='ETTh1', save_every=5, eval_every=5, 
                                patience=6, pred_len=96):
    """
    Training function with early stopping and periodic evaluation
    """
    
    # Initialize components
    tokenizer = build_tokenizer(
        quant_range=quant_range,
        vocab_size=vocab_size,
        context_length=seq_len,
        prediction_length=pred_len
    )
    
    early_stopping = EarlyStopping(patience=patience, min_delta=1e-6)
    logger = TrainingLogger(output_dir, dataset_name)
    
    num_batches = len(train_loader)
    total_steps = epochs * num_batches
    min_lr = learning_rate * min_lr_factor
    best_val_loss = float('inf')
    
    print(f"\n🚀 Starting training with early stopping...")
    print(f"📝 Configuration:")
    print(f"   Max epochs: {epochs}")
    print(f"   Early stopping patience: {patience}")
    print(f"   Evaluation every: {eval_every} epochs")
    print(f"   Save every: {save_every} epochs")
    print(f"   Prediction length for eval: {pred_len}")
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        t1 = time.time()
        epoch_loss = 0
        current_lr = 0
        
        progress_bar = tqdm(
            enumerate(train_loader), 
            total=len(train_loader), 
            desc=f"🏃 Epoch {epoch+1}/{epochs}", 
            position=0, 
            leave=True
        )
        
        for i, (batch_x, batch_y, _, _) in progress_bar:
            iteration = epoch * num_batches + i
            x = batch_x.float().squeeze(-1)
            y = batch_y.float().squeeze(-1)
            
            # Get learning rate
            lr = get_lr(iteration, total_steps, warmup_steps, learning_rate, min_lr, decay_lr)
            current_lr = lr
            
            # Update learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
            total_loss = 0
            optimizer.zero_grad(set_to_none=True)
            
            # Gradient accumulation loop
            for micro_step in range(grad_accumulation_steps):
                token_ids, attention_mask, tokenizer_state = tokenizer.context_input_transform(x)
                # target_token_ids, target_attention_mask = tokenizer.label_input_transform(y, tokenizer_state)
                
                # Forward pass
                logits = model(token_ids.to(device), patch_lengths)
                
                # Calculate loss
                loss = F.mse_loss(logits, y.to(device), reduction='mean')
                # loss = F.cross_entropy(
                #     logits.reshape(-1, logits.size(-1)),
                #     target_token_ids.reshape(-1).to(device),
                #     ignore_index=PAD_ID
                # )
                loss = loss / grad_accumulation_steps
                
                # Backward pass
                scaler.scale(loss).backward()
                total_loss += loss.item() * grad_accumulation_steps
            
            # Gradient clipping
            if clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                
            # Update weights
            scaler.step(optimizer)
            scaler.update()
            
            # Update metrics
            epoch_loss += total_loss
            avg_epoch_loss = epoch_loss / (i + 1)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{total_loss:.4f}",
                'avg_loss': f"{avg_epoch_loss:.4f}",
                'lr': f"{lr:.6f}",
                'patience': f"{early_stopping.counter}/{patience}"
            })
        
        # Calculate training metrics
        train_time = time.time() - t1
        train_avg_loss = epoch_loss / len(train_loader)
        
        # Validation phase
        print(f"\n🔍 Running validation for epoch {epoch+1}...")
        t1 = time.time()
        model.eval()
        val_loss = validate(model, validate_loader, tokenizer, patch_lengths, device, 
                          desc=f"Epoch {epoch+1} Validation")
        val_time = time.time() - t1
        
        # Print epoch results
        print(f"\n📊 Epoch {epoch+1}/{epochs} Results:")
        print(f"   Training Loss: {train_avg_loss:.6f} (Time: {train_time:.2f}s)")
        print(f"   Validation Loss: {val_loss:.6f} (Time: {val_time:.2f}s)")
        print(f"   Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler': scaler.state_dict() if scaler else None,
                'val_loss': val_loss,
                'train_loss': train_avg_loss
            }, os.path.join(output_dir, f'best_model_{dataset_name}_{seq_len}.pth'))
            print(f"   ✅ New best model saved! (Val Loss: {best_val_loss:.6f})")
        
        # Periodic evaluation
        eval_results = None
        if (epoch + 1) % eval_every == 0:
            print(f"\n🎯 Running full evaluation at epoch {epoch+1}...")
            try:
                eval_results = evaluation(
                    model, 
                    dataset_name, 
                    features,
                    quant_range,
                    vocab_size,
                    input_len=seq_len,
                    pred_len=pred_len,
                    eval_batch_size=batch_size,
                    device=device
                )
                print(f"   📈 Evaluation completed successfully!")
            except Exception as e:
                print(f"   ❌ Evaluation failed: {e}")
                eval_results = None
        
        # Log metrics
        logger.log_epoch(epoch + 1, train_avg_loss, val_loss, current_lr, eval_results)
        
        # Save periodic checkpoint
        if save_every > 0 and (epoch + 1) % save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler': scaler.state_dict() if scaler else None,
                'val_loss': val_loss,
                'train_loss': train_avg_loss
            }, os.path.join(output_dir, f'checkpoint_{seq_len}_epoch_{epoch+1}.pth'))
            print(f"   💾 Checkpoint saved at epoch {epoch+1}")
        
        # Early stopping check
        if early_stopping(val_loss, model):
            print(f"\n🛑 Early stopping triggered after {epoch+1} epochs!")
            print(f"   No improvement for {patience} consecutive epochs")
            print(f"   Best validation loss: {early_stopping.best_loss:.6f}")
            break
    
    # Training completed
    print(f"\n🎉 Training completed!")
    
    # Final evaluation if not recently done
    if (epoch + 1) % eval_every != 0:
        print(f"\n🎯 Running final evaluation...")
        try:
            final_eval_results = evaluation(
                model, 
                dataset_name, 
                features,
                quant_range,
                vocab_size,
                input_len=seq_len,
                pred_len=pred_len,
                eval_batch_size=batch_size,
                device=device
            )
            logger.log_epoch(epoch + 1, train_avg_loss, val_loss, current_lr, final_eval_results)
        except Exception as e:
            print(f"   ❌ Final evaluation failed: {e}")
    
    # Print summary
    logger.print_summary()
    
    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler': scaler.state_dict() if scaler else None,
        'val_loss': val_loss,
        'train_loss': train_avg_loss,
        'early_stopped': early_stopping.counter >= patience
    }, os.path.join(output_dir, f'final_model_{dataset_name}_{seq_len}.pth'))
    
    return model, logger.history

## ------------------------------
# Main Training and Evaluation
## ------------------------------

if __name__ == "__main__":
    pred_len = 96  # Set prediction length for evaluation
    
    print(f"\n{'='*80}")
    print(f"🤖 BYTE LATENT TRANSFORMER TRAINING")
    print(f"{'='*80}")
    print(f"Model parameters: {count_parameters(model) / 1e6:.2f}M")
    print(f"Dataset: {dataset_name} ({features} features)")
    print(f"Sequence length: {seq_len}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    print(f"Precision: {dtype}")
    
    # Start training
    trained_model, training_history = training_with_early_stopping(
        model=model,
        train_loader=train_loader,
        validate_loader=validate_loader,
        patch_lengths=patch_lengths,
        device=device,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        grad_accumulation_steps=grad_accumulation_steps,
        clip_grad=clip_grad,
        warmup_steps=warmup_steps,
        min_lr_factor=min_lr_factor,
        decay_lr=decay_lr,
        output_dir=output_dir,
        dataset_name=dataset_name,
        save_every=save_every,
        eval_every=eval_every,
        patience=patience,
        pred_len=pred_len
    )
    
    print(f"\n✅ Training pipeline completed successfully!")
    print(f"📁 All outputs saved to: {output_dir}")