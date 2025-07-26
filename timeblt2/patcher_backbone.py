import warnings
warnings.filterwarnings("ignore")
from bytelatent.model.blt import ByteLatentTransformerArgs, ByteLatentTransformer

## Training Args
vocab_size = 2048
quant_range = 10
batch_size = 512
seq_len = 96


## Set model args
model_args = ByteLatentTransformerArgs(
    seed=42,
    vocab_size=vocab_size,                       # Small byte-level vocab
    max_length=seq_len,                        # Max full sequence length
    max_seqlen=seq_len,
    max_encoder_seq_length=seq_len,
    local_attention_window_len=seq_len,        # Local window, 128 is sufficient for small models

    dim_global=64,                        # Lower than default 512
    dim_local_encoder=32,
    dim_local_decoder=32,

    n_layers_global=3,
    n_layers_local_encoder=3,
    n_layers_local_decoder=3,

    n_heads_global=8,                      # Reduce heads
    n_heads_local_encoder=4,
    n_heads_local_decoder=4,

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

    encoder_hash_byte_group_size=[4,5,6],   # Fewer hash sizes
    encoder_hash_byte_group_vocab=2**6,
    encoder_hash_byte_group_nb_functions=1,
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
