#!/bin/bash

# Create log directories
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

if [ ! -d "./logs/LongForecasting/ParamSweep" ]; then
    mkdir ./logs/LongForecasting/ParamSweep
fi

# Dataset configuration
model_name=PatchTST
root_path_name=./dataset/
data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1
random_seed=2021
pred_len=96

echo "Starting ETTm1 Parameter Sweep Experiments"
echo "Prediction length: $pred_len"
echo "=================================================="

# Function to run experiment
run_experiment() {
    local seq_len=$1
    local batch_size=$2
    local patching_batch_size=$3
    local max_patch_length=$4
    local n_layers_local=$5
    local n_layers_global=$6
    local dim_global=$7
    local dim_local=$8
    local cross_attn_k=$9
    local learning_rate=${10}
    local train_epochs=${11}
    local patience=${12}
    local dropout=${13}
    local patching_threshold=${14}
    local patching_threshold_add=${15}
    local quant_range=${16}
    local config_name=${17}
    
    # Calculate derived parameters based on constraints
    local d_ff=$((dim_global * 4))
    local dim_local_encoder=$dim_local
    local dim_local_decoder=$dim_local
    
    # Calculate number of heads as factors of dimensions
    local n_heads_global=4
    if [ $dim_global -eq 32 ]; then
        n_heads_global=4  # 32/4=8
    elif [ $dim_global -eq 48 ]; then
        n_heads_global=4  # 48/4=12
    elif [ $dim_global -eq 64 ]; then
        n_heads_global=4  # 64/4=16
    elif [ $dim_global -eq 80 ]; then
        n_heads_global=4  # 80/4=20
    elif [ $dim_global -eq 96 ]; then
        n_heads_global=4  # 96/4=24
    fi
    
    local n_heads_local=4
    if [ $dim_local -eq 16 ]; then
        n_heads_local=4  # 16/4=4
    elif [ $dim_local -eq 24 ]; then
        n_heads_local=4  # 24/4=6
    elif [ $dim_local -eq 32 ]; then
        n_heads_local=4  # 32/4=8
    elif [ $dim_local -eq 40 ]; then
        n_heads_local=4  # 40/4=10
    fi
    
    # Fixed cross attention heads
    local cross_attn_nheads=2
    
    # Create unique model ID
    local current_model_id="${model_id_name}_${seq_len}_${pred_len}_${config_name}"
    
    echo ""
    echo "Running: $config_name"
    echo "-------------------------------------------"
    echo "Seq Len: $seq_len | Batch: $batch_size | Max Patch: $max_patch_length"
    echo "Local Layers: $n_layers_local | Global Layers: $n_layers_global"
    echo "Global Dim: $dim_global (heads: $n_heads_global) | Local Dim: $dim_local (heads: $n_heads_local)"
    echo "Cross Attn K: $cross_attn_k | Quant Range: $quant_range | LR: $learning_rate"
    echo "Thresholds: $patching_threshold + $patching_threshold_add | Dropout: $dropout"
    
    # Run the experiment
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $current_model_id \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --vocab_size 256 \
      --quant_range $quant_range \
      --n_layers_local_encoder $n_layers_local \
      --n_layers_local_decoder $n_layers_local \
      --n_layers_global $n_layers_global \
      --n_heads_local_encoder $n_heads_local \
      --n_heads_local_decoder $n_heads_local \
      --n_heads_global $n_heads_global \
      --dim_global $dim_global \
      --dim_local_encoder $dim_local_encoder \
      --dim_local_decoder $dim_local_decoder \
      --cross_attn_k $cross_attn_k \
      --cross_attn_nheads $cross_attn_nheads \
      --d_ff $d_ff \
      --dropout $dropout \
      --fc_dropout $dropout \
      --head_dropout $dropout \
      --max_patch_length $max_patch_length \
      --patching_threshold $patching_threshold \
      --patching_threshold_add $patching_threshold_add \
      --monotonicity 1 \
      --des 'ParamSweep' \
      --train_epochs $train_epochs \
      --patience $patience \
      --lradj 'TST' \
      --pct_start 0.4 \
      --itr 1 \
      --batch_size $batch_size \
      --patching_batch_size $patching_batch_size \
      --learning_rate $learning_rate \
      >logs/LongForecasting/ParamSweep/${model_name}_${current_model_id}.log 2>&1
    
    # Check result
    if [ $? -eq 0 ]; then
        echo "✓ SUCCESS: $config_name"
    else
        echo "✗ FAILED: $config_name"
        echo "Check log: logs/LongForecasting/ParamSweep/${model_name}_${current_model_id}.log"
    fi
}

# Run all configurations
echo ""
echo "LONG SEQUENCE EXPERIMENTS (seq_len=336)"
echo "========================================"

# Config 1: baseline_small - Conservative baseline
# cross_attn_k=2 means dim_global=2*dim_local (32=2*16)
run_experiment 96 128 336 16 2 2 32 16 2 0.0001 100 20 0.05 0.3 0.2 6 "baseline_small"
sleep 2

# Config 2: baseline_medium - Medium capacity
# cross_attn_k=2 means dim_global=2*dim_local (48=2*24)
run_experiment 96 128 336 16 2 3 48 24 2 0.00008 120 22 0.05 0.35 0.25 8 "baseline_medium"
sleep 2

# Config 3: baseline_large - Larger model
# cross_attn_k=2 means dim_global=2*dim_local (64=2*32)
run_experiment 96 128 336 16 3 3 64 32 2 0.00006 140 25 0.04 0.4 0.3 6 "baseline_large"
sleep 2

# Config 4: wide_architecture - Wider dimensions
# cross_attn_k=3 means dim_global=3*dim_local (96=3*32)
run_experiment 96 128 336 16 2 2 96 32 3 0.00005 130 25 0.06 0.32 0.22 8 "wide_architecture"
sleep 2

# Config 5: deep_local - More local layers
# cross_attn_k=2 means dim_global=2*dim_local (64=2*32)
run_experiment 96 128 336 12 4 2 64 32 2 0.00007 120 20 0.05 0.28 0.18 12 "deep_local"
sleep 2

# Config 6: deep_global - More global layers
# cross_attn_k=2 means dim_global=2*dim_local (48=2*24)
run_experiment 96 128 336 16 2 4 48 24 2 0.00008 130 25 0.04 0.35 0.25 6 "deep_global"
sleep 2

# Config 7: fine_patches - Small max patches, low thresholds
# cross_attn_k=2.5 means dim_global=2.5*dim_local (40=2.5*16)
run_experiment 96 128 336 8 2 3 40 16 2 0.00009 110 20 0.06 0.2 0.15 8 "fine_patches"
sleep 2

# Config 8: coarse_patches - Large max patches, high thresholds
# cross_attn_k=2 means dim_global=2*dim_local (64=2*32)
run_experiment 96 128 336 20 2 2 64 32 2 0.00006 140 30 0.05 0.6 0.4 4 "coarse_patches"
sleep 2

# Config 9: high_quant - Higher quantization range
# cross_attn_k=2 means dim_global=2*dim_local (48=2*24)
run_experiment 96 128 336 16 3 3 48 24 2 0.00007 125 22 0.05 0.35 0.25 12 "high_quant"
sleep 2

# # Config 10: low_quant - Lower quantization range
# # cross_attn_k=2 means dim_global=2*dim_local (32=2*16)
# run_experiment 96 128 336 12 2 2 32 16 2 0.0001 110 20 0.06 0.3 0.2 4 "low_quant"
# sleep 2

# # Config 11: high_reg - Heavy regularization
# # cross_attn_k=2 means dim_global=2*dim_local (48=2*24)
# run_experiment 96 128 336 16 3 3 48 24 2 0.00008 120 18 0.1 0.25 0.15 6 "high_reg"
# sleep 2

# # Config 12: low_reg - Light regularization
# # cross_attn_k=2 means dim_global=2*dim_local (64=2*32)
# run_experiment 96 128 336 16 2 2 64 32 2 0.00012 100 25 0.02 0.45 0.35 8 "low_reg"
# sleep 2

# # Config 13: balanced_deep - Balanced deep architecture
# # cross_attn_k=2 means dim_global=2*dim_local (56=2*28), moderate depth
# run_experiment 96 96 288 14 3 3 56 28 2 0.00006 135 24 0.055 0.33 0.23 8 "balanced_deep"
# sleep 2

# # Config 14: ultra_wide - Maximum width experiment
# # cross_attn_k=4 means dim_global=4*dim_local (128=4*32)
# run_experiment 96 64 192 16 2 2 128 32 4 0.00003 160 30 0.04 0.38 0.28 6 "ultra_wide"
# sleep 2

# # Config 15: minimal_arch - Minimal but efficient
# # cross_attn_k=2 means dim_global=2*dim_local (24=2*12)
# run_experiment 96 160 480 10 1 1 24 12 2 0.00015 80 18 0.07 0.25 0.15 4 "minimal_arch"
# sleep 2

# # Config 16: hybrid_depth - Mixed depth strategy
# # cross_attn_k=2.5 means dim_global=2.5*dim_local (80=2.5*32)
# run_experiment 96 80 240 16 5 3 80 32 2 0.00004 150 28 0.045 0.4 0.3 10 "hybrid_depth"
# sleep 2

# # Config 17: adaptive_patches - Moderate adaptive patching
# # cross_attn_k=2 means dim_global=2*dim_local (60=2*30)
# run_experiment 96 112 336 14 3 2 60 30 2 0.00008 115 22 0.05 0.35 0.27 8 "adaptive_patches"
# sleep 2

# # Config 18: memory_efficient - Optimized for memory
# # cross_attn_k=2 means dim_global=2*dim_local (48=2*24)
# run_experiment 96 96 288 12 2 2 48 24 2 0.00009 120 25 0.06 0.45 0.32 6 "memory_efficient"
# sleep 2

# # Config 19: high_capacity - Maximum model capacity
# # cross_attn_k=3 means dim_global=3*dim_local (120=3*40)
# run_experiment 96 48 144 18 4 4 120 40 3 0.000025 200 40 0.03 0.42 0.32 12 "high_capacity"
# sleep 2

# # Config 20: fast_convergence - Optimized for quick training
# # cross_attn_k=2 means dim_global=2*dim_local (40=2*20)
# run_experiment 96 144 432 12 2 2 40 20 2 0.00012 75 15 0.08 0.28 0.18 6 "fast_convergence"
# sleep 2

# # Config 21: precision_focus - High precision quantization
# # cross_attn_k=2 means dim_global=2*dim_local (64=2*32)
# run_experiment 96 80 240 16 3 3 64 32 2 0.00005 140 25 0.04 0.38 0.28 16 "precision_focus"
# sleep 2

# # Config 22: ultra_fine - Ultra fine-grained patching
# # cross_attn_k=2 means dim_global=2*dim_local (32=2*16)
# run_experiment 96 128 384 6 2 3 32 16 2 0.0001 110 20 0.065 0.15 0.1 8 "ultra_fine"
# sleep 2

# # Config 23: ultra_coarse - Ultra coarse-grained patching
# # cross_attn_k=2 means dim_global=2*dim_local (80=2*40)
# run_experiment 33966 72 216 24 2 2 80 40 2 0.00004 150 35 0.045 0.7 0.5 4 "ultra_coarse"
# sleep 2

# # Config 24: robust_training - Robust with high patience
# # cross_attn_k=2 means dim_global=2*dim_local (56=2*28)
# run_experiment 96 96 288 16 3 3 56 28 2 0.00006 180 45 0.055 0.35 0.25 8 "robust_training"
# sleep 2

echo ""
echo "LONG SEQUENCE EXPERIMENTS (seq_len=336)" 
echo "========================================"

echo ""
echo "=================================================="
echo "PARAMETER SWEEP SUMMARY"
echo "=================================================="
echo ""
echo "Long Sequence (336) Configurations - 24 Variants:"
echo "1.  baseline_small      - Conservative (32/16, k=2)"
echo "2.  baseline_medium     - Medium capacity (48/24, k=2)"
echo "3.  baseline_large      - Larger model (64/32, k=2)"
echo "4.  wide_architecture   - Wide dims (96/32, k=3)"
echo "5.  deep_local          - More local layers (4 local, 2 global)"
echo "6.  deep_global         - More global layers (2 local, 4 global)"
echo "7.  fine_patches        - Small patches (max=8, thresh=0.2+0.15)"
echo "8.  coarse_patches      - Large patches (max=20, thresh=0.6+0.4)"
echo "9.  high_quant          - High quantization (range=12)"
echo "10. low_quant           - Low quantization (range=4)"
echo "11. high_reg            - Heavy regularization (dropout=0.1)"
echo "12. low_reg             - Light regularization (dropout=0.02)"
echo "13. balanced_deep       - Balanced 3+3 layers (56/28)"
echo "14. ultra_wide          - Maximum width (128/32, k=4)"
echo "15. minimal_arch        - Minimal efficient (24/12, 1+1 layers)"
echo "16. hybrid_depth        - Mixed depth (5 local, 3 global)"
echo "17. adaptive_patches    - Moderate adaptive (60/30)"
echo "18. memory_efficient    - Memory optimized (48/24)"
echo "19. high_capacity       - Maximum capacity (120/40, 4+4 layers)"
echo "20. fast_convergence    - Quick training (75 epochs)"
echo "21. precision_focus     - High precision (quant=16)"
echo "22. ultra_fine          - Ultra fine patches (max=6, thresh=0.15+0.1)"
echo "23. ultra_coarse        - Ultra coarse patches (max=24, thresh=0.7+0.5)"
echo "24. robust_training     - High patience (180 epochs, patience=45)"

# Create summary
echo ""
echo "=================================================="
echo "PARAMETER SWEEP SUMMARY"
echo "=================================================="
echo ""
echo "Short Sequence (96) Configurations:"
echo "1.  baseline_small     - Conservative (dim_global=32, dim_local=16, k=2)"
echo "2.  baseline_medium    - Medium capacity (dim_global=48, dim_local=24, k=2)"
echo "3.  baseline_large     - Larger model (dim_global=64, dim_local=32, k=2)"
echo "4.  wide_architecture  - Wide dims (dim_global=96, dim_local=32, k=3)"
echo "5.  deep_local         - More local layers (4 local, 2 global)"
echo "6.  deep_global        - More global layers (2 local, 4 global)"
echo "7.  fine_patches       - Small patches (max_len=8, thresh=0.2+0.15)"
echo "8.  coarse_patches     - Large patches (max_len=20, thresh=0.6+0.4)"
echo "9.  high_quant         - High quantization (quant_range=12)"
echo "10. low_quant          - Low quantization (quant_range=4)"
echo "11. high_reg           - Heavy regularization (dropout=0.1)"
echo "12. low_reg            - Light regularization (dropout=0.02)"
echo ""
echo "Long Sequence (336) Configurations:"
echo "13. long_baseline      - Conservative for long sequences"
echo "14. long_deep          - Deep architecture (4+4 layers)"
echo "15. long_wide          - Wide architecture (dim_global=96)"
echo "16. long_efficient     - Efficient coarse patching"
echo "17. long_fine          - Fine-grained long sequences"
echo ""
echo "Key Parameter Relationships:"
echo "- cross_attn_k: Ratio of global_dim to local_dim (2, 2.5, 3, or 4)"
echo "- Local encoder/decoder: Same layers and dimensions"
echo "- Heads: Always factors of respective dimensions (typically 4)"
echo "- Cross attention heads: Fixed at 2"
echo "- Vocab size: Fixed at 256"
echo "- Quantization range: 4, 6, 8, 10, 12, 16"
echo "- Max patch length: 6, 8, 10, 12, 14, 16, 18, 20, 24"
echo "- Patch sizes: Auto-adapted based on thresholds"
echo "- Batch size: Fixed at 128, patching_batch_size: Fixed at 336"
echo ""
echo "Architecture Strategies Tested:"
echo "- Baseline: Conservative balanced architectures"
echo "- Deep: More layers for complex pattern learning"
echo "- Wide: Larger dimensions for representation capacity"
echo "- Minimal: Efficient lightweight architectures"
echo "- Hybrid: Mixed local/global processing strategies"
echo "- Patch-focused: Different temporal granularity approaches"
echo "- Training-focused: Different convergence and regularization strategies"
echo ""
echo "Check logs in: logs/LongForecasting/ParamSweep/"
echo "Monitor results in wandb dashboard"
echo ""

# Generate results comparison script
cat > analyze_results.py << 'EOF'
#!/usr/bin/env python3
"""
Quick script to analyze parameter sweep results
"""
import os
import re

def parse_log_file(filepath):
    """Extract key metrics from log file"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Extract final test metrics
        mse_match = re.search(r'mse:([\d.]+)', content)
        mae_match = re.search(r'mae:([\d.]+)', content)
        rse_match = re.search(r'rse:([\d.]+)', content)
        
        mse = float(mse_match.group(1)) if mse_match else None
        mae = float(mae_match.group(1)) if mae_match else None  
        rse = float(rse_match.group(1)) if rse_match else None
        
        return {'mse': mse, 'mae': mae, 'rse': rse}
    except:
        return None

def main():
    log_dir = "logs/LongForecasting/ParamSweep/"
    
    if not os.path.exists(log_dir):
        print("No parameter sweep logs found!")
        return
    
    results = []
    
    for filename in os.listdir(log_dir):
        if filename.endswith('.log'):
            filepath = os.path.join(log_dir, filename)
            metrics = parse_log_file(filepath)
            
            if metrics and metrics['mse'] is not None:
                config_name = filename.replace('PatchTST_ETTm1_', '').replace('.log', '')
                results.append({
                    'config': config_name,
                    'filename': filename,
                    **metrics
                })
    
    if not results:
        print("No completed experiments found!")
        return
    
    # Sort by MSE (lower is better)
    results.sort(key=lambda x: x['mse'])
    
    print("\nPARAMETER SWEEP RESULTS (sorted by MSE)")
    print("="*75)
    print(f"{'Rank':<4} {'Configuration':<30} {'MSE':<8} {'MAE':<8} {'RSE':<8}")
    print("-"*75)
    
    for i, result in enumerate(results, 1):
        print(f"{i:<4} {result['config']:<30} {result['mse']:<8.4f} {result['mae']:<8.4f} {result['rse']:<8.4f}")
    
    print(f"\nBest configuration: {results[0]['config']}")
    print(f"Best MSE: {results[0]['mse']:.4f}")
    
    # Show top 5
    print(f"\nTOP 5 CONFIGURATIONS:")
    for i in range(min(5, len(results))):
        print(f"{i+1}. {results[i]['config']} - MSE: {results[i]['mse']:.4f}")
    
    # Analyze by sequence length
    short_seq = [r for r in results if r['config'].startswith('96_96_')]
    long_seq = [r for r in results if r['config'].startswith('336_96_')]
    
    if short_seq:
        print(f"\nBest Short Sequence (96): {short_seq[0]['config']} - MSE: {short_seq[0]['mse']:.4f}")
    if long_seq:
        print(f"Best Long Sequence (336): {long_seq[0]['config']} - MSE: {long_seq[0]['mse']:.4f}")
    
    # Analyze by architecture type
    print(f"\nArchitecture Analysis:")
    baseline_configs = [r for r in results if 'baseline' in r['config']]
    deep_configs = [r for r in results if 'deep' in r['config']]
    wide_configs = [r for r in results if 'wide' in r['config']]
    
    if baseline_configs:
        best_baseline = min(baseline_configs, key=lambda x: x['mse'])
        print(f"Best Baseline: {best_baseline['config']} - MSE: {best_baseline['mse']:.4f}")
    
    if deep_configs:
        best_deep = min(deep_configs, key=lambda x: x['mse'])
        print(f"Best Deep: {best_deep['config']} - MSE: {best_deep['mse']:.4f}")
        
    if wide_configs:
        best_wide = min(wide_configs, key=lambda x: x['mse'])
        print(f"Best Wide: {best_wide['config']} - MSE: {best_wide['mse']:.4f}")

if __name__ == "__main__":
    main()
EOF

chmod +x analyze_results.py

echo "Run 'python analyze_results.py' after experiments complete to see rankings!"
echo "=================================================="