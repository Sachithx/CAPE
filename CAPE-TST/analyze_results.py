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
