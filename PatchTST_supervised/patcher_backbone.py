import warnings
warnings.filterwarnings("ignore")
from bytelatent.model.blt import ByteLatentTransformerArgs, ByteLatentTransformer
from utils.patch_utils import build_tokenizer


## Training Args
vocab_size = 256
quant_range = 5
seq_len = 336

# Initialize components
tokenizer = build_tokenizer(
    quant_range=quant_range,
    vocab_size=vocab_size,
    context_length=seq_len,
    prediction_length=seq_len
)

## Set model args
model_args = ByteLatentTransformerArgs(
    seed=42,
    vocab_size=vocab_size,                       # Small byte-level vocab
    max_length=seq_len,                        # Max full sequence length
    max_seqlen=seq_len,
    max_encoder_seq_length=seq_len,
    local_attention_window_len=seq_len,        # Local window, 128 is sufficient for small models

    dim_global=64,                        # Lower than default 512
    dim_local_encoder=64,
    dim_local_decoder=64,

    n_layers_global=2,
    n_layers_local_encoder=2,
    n_layers_local_decoder=2,

    n_heads_global=4,                      # Reduce heads
    n_heads_local_encoder=4,
    n_heads_local_decoder=4,

    patch_size=8,
    patch_in_forward=True,                # Patch in forward pass
    patching_batch_size=512,
    patching_device="cuda",               # Use CPU for patching in small model
    patching_mode="entropy",
    patching_threshold=0.3,
    patching_threshold_add=0.2,           # No additional threshold
    max_patch_length=8,
    monotonicity=True,            # Monotonic patching
    pad_to_max_length=True,

    cross_attn_encoder=True,
    cross_attn_decoder=True,
    cross_attn_k=1,
    cross_attn_nheads=2,
    cross_attn_all_layers_encoder=True,
    cross_attn_all_layers_decoder=True,
    cross_attn_use_flex_attention=False,
    cross_attn_init_by_pooling=True,

    encoder_hash_byte_group_size=[10],   # Fewer hash sizes
    encoder_hash_byte_group_vocab=2**4,
    encoder_hash_byte_group_nb_functions=2,
    encoder_enable_byte_ngrams=False,

    non_linearity="swiglu",
    use_rope=True,
    attn_impl="sdpa",                      # Efficient PyTorch attention
    attn_bias_type="causal",

    dropout=0.0,
    layer_ckpt="none",                     # No checkpointing in small model
    init_use_gaussian=True,
    init_use_depth="current",
    alpha_depth="disabled",
    log_patch_lengths=True,

    downsampling_by_pooling="max",         # Efficient downsampling
    use_local_encoder_transformer=True,
    share_encoder_decoder_emb=False         # Save memory if possible
)

patcher_backbone = ByteLatentTransformer(model_args)
