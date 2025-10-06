#!/usr/bin/env python3
"""
Ablation Study Script Generator for PatchTST
Generates commands for systematic threshold ablation study
"""

import itertools

# Base configuration from your best performing model (3rd command)
base_config = {
    "random_seed": 2025,
    "is_training": 1,
    "root_path": "./dataset/",
    "entropy_model_checkpoint_dir": "./entropy_model_checkpoints/",
    "data_path": "ETTm1.csv",
    "model_id": "ETTm1_96_96",
    "model_id_name": "ETTm1",
    "model": "PatchTST",
    "data": "ETTm1",
    "features": "M",
    "seq_len": 96,
    "pred_len": 96,
    "enc_in": 7,
    "vocab_size": 256,
    "quant_range": 3,
    "n_layers_local_encoder": 1,
    "n_layers_local_decoder": 1,
    "n_layers_global": 1,
    "dim_global": 16,
    "dim_local_encoder": 16,
    "dim_local_decoder": 16,
    "cross_attn_k": 1,
    "n_heads_local_encoder": 8,
    "n_heads_local_decoder": 8,
    "n_heads_global": 8,
    "cross_attn_nheads": 8,
    "cross_attn_window_encoder": 96,
    "cross_attn_window_decoder": 96,
    "local_attention_window_len": 96,
    "dropout": 0.2,
    "multiple_of": 32,
    "monotonicity": 1,
    "des": "Exp",
    "train_epochs": 30,
    "patience": 7,
    "lradj": "TST",
    "pct_start": 0.3,
    "itr": 1,
    "batch_size": 32,
    "patching_batch_size": 2048,
    "learning_rate": 0.01
}

# Fixed ablation parameters
PATCH_SIZE = 24
MAX_PATCH_LENGTH = 24

# Threshold ranges for ablation
patching_thresholds = [0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0, 3.3, 3.6, 3.9]
patching_threshold_adds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

def generate_command(config, patch_threshold, patch_threshold_add, experiment_id):
    """Generate a single command with specified parameters"""
    # Update config with ablation parameters
    updated_config = config.copy()
    updated_config.update({
        "patch_size": PATCH_SIZE,
        "max_patch_length": MAX_PATCH_LENGTH,
        "patching_threshold": patch_threshold,
        "patching_threshold_add": patch_threshold_add,
        "model_id": f"ablation_{experiment_id}_{PATCH_SIZE}_{patch_threshold}_{patch_threshold_add}"
    })
    
    # Build command string
    cmd_parts = ["python run_longExp.py"]
    for key, value in updated_config.items():
        cmd_parts.append(f"--{key} {value}")
    
    return " ".join(cmd_parts)

def generate_all_commands():
    """Generate all ablation study commands"""
    commands = []
    experiment_id = 1
    
    print("# PatchTST Ablation Study Commands")
    print(f"# Fixed: patch_size={PATCH_SIZE}, max_patch_length={MAX_PATCH_LENGTH}")
    print(f"# Total experiments: {len(patching_thresholds) * len(patching_threshold_adds)}")
    print("# Format: patching_threshold x patching_threshold_add\n")
    
    for threshold in patching_thresholds:
        for threshold_add in patching_threshold_adds:
            cmd = generate_command(
                base_config, 
                threshold, 
                threshold_add, 
                experiment_id
            )
            
            print(f"# Experiment {experiment_id}: threshold={threshold}, threshold_add={threshold_add}")
            print(cmd)
            print()
            
            commands.append({
                'id': experiment_id,
                'threshold': threshold,
                'threshold_add': threshold_add,
                'command': cmd
            })
            experiment_id += 1
    
    return commands

def generate_batch_script():
    """Generate a bash script to run all experiments"""
    print("\n" + "="*80)
    print("BASH SCRIPT VERSION")
    print("="*80)
    print("#!/bin/bash")
    print("# Ablation Study Batch Script")
    print(f"# Total experiments: {len(patching_thresholds) * len(patching_threshold_adds)}")
    print()
    
    experiment_id = 1
    for threshold in patching_thresholds:
        for threshold_add in patching_threshold_adds:
            cmd = generate_command(
                base_config, 
                threshold, 
                threshold_add, 
                experiment_id
            )
            
            print(f"echo 'Running experiment {experiment_id}: threshold={threshold}, threshold_add={threshold_add}'")
            print(cmd)
            print(f"echo 'Completed experiment {experiment_id}'")
            print()
            experiment_id += 1

def generate_results_table():
    """Generate template for results tracking"""
    print("\n" + "="*80)
    print("RESULTS TRACKING TABLE")
    print("="*80)
    print("exp_id,patch_size,max_patch_length,patching_threshold,patching_threshold_add,mse_loss,mae_loss,training_time")
    
    experiment_id = 1
    for threshold in patching_thresholds:
        for threshold_add in patching_threshold_adds:
            print(f"{experiment_id},{PATCH_SIZE},{MAX_PATCH_LENGTH},{threshold},{threshold_add},,,")
            experiment_id += 1

if __name__ == "__main__":
    # Generate individual commands
    commands = generate_all_commands()
    
    # Generate batch script
    generate_batch_script()
    
    # Generate results tracking template
    generate_results_table()
    
    print(f"\nTotal experiments generated: {len(commands)}")
    print(f"Threshold values: {len(patching_thresholds)} values from {min(patching_thresholds)} to {max(patching_thresholds)}")
    print(f"Threshold_add values: {len(patching_threshold_adds)} values from {min(patching_threshold_adds)} to {max(patching_threshold_adds)}")