import json
import os
import random
import sys

sys.path.insert(0, '..')  # find dglgcn.py when run from OUTPUTS/

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
import torch

from dglgcn import (
    Config,
    GCN,
    Xnoised_dataset,
    Ynoised_dataset,
    dgldataset,
    compute_threshold_from_split,
    omit_sensitive_data,
    evaluate,
    local_loss,
    local_spearman,
    set_seeds,
    train as gcn_train,
)

OUTPUT_DIR = 'OUTPUTS/adversarial'
ADVERSARIAL_SEED = 555
CENSOR_REGION = 'above'

# sweep constants
NUM_TRIALS = 5
CENSOR_SPLITS = [0.1, 0.5, 0.9]
CLEAN_FRACS = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
NOISE_TYPES = ['ynoise', 'xnoise', 'omission']
YNOISE_LEVELS = [0.0, 0.5, 2.0, 5.0]
XNOISE_RANGES = [[1.0, 1.0], [0.8, 1.0], [0.3, 0.4], [0, 0.05]]
OMIT_FRACTIONS = [0.0, 0.5, 1.0]

MODEL_CONFIG = Config(    # matches main notebooks
    lr=0.005,
    hdim=128,
    epochs=200,
    patience=10,
    min_delta=1e-4,
    batch_size=32,
    loss_func='mse',
)

FINETUNE_CONFIG = Config(
    lr=0.001,
    hdim=128,
    epochs=150,
    patience=10,
    min_delta=1e-4,
    batch_size=32,
    loss_func='mse',
)

def adversarial_retrain_wrapper(
    noise_type,
    noise_param,
    censor_region,
    censor_split,
    clean_frac,
    model_config=None,
    finetune_config=None,
    jobname=None,
    dir_name=None,
    random_state=None,
    verbose=True,
    rawdata=None,
):
    """
    Full adversarial retraining experiment for one configuration.

    Steps:
        1. Split data fresh (adversary has no knowledge of original splits).
        2. Apply noise to train+val, train GCN on noised data.
        3. From the clean train data, isolate the sensitive region and
           subsample to clean_frac.
        4. Fine-tune the whole model on that clean sensitive subset.
        5. Evaluate on the clean test set.

    Args:
        noise_type   : 'ynoise', 'xnoise', or 'omission'
        noise_param  : mag_noise (float) for ynoise,
                       simscore_range (list) for xnoise,
                       omit_fraction (float) for omission
        censor_region: 'above' or 'below'
        censor_split : fraction of dataset that is sensitive
        clean_frac   : fraction of clean sensitive train data available to adversary
        model_config : Config for initial noised training
        finetune_config: Config for fine-tuning pass
        jobname      : string identifier for output files
        dir_name     : output directory
        random_state : seed
        verbose      : print progress
        rawdata      : pd.DataFrame with columns ['smiles', 'labels']

    Returns:
        dict with keys:
            noised_*   — metrics after training on noised data (pre-finetune)
            finetuned_* — metrics after fine-tuning on clean sensitive subset
    """
    if rawdata is None:
        raise ValueError("rawdata must be provided.")
    if model_config is None:
        model_config = MODEL_CONFIG
    if finetune_config is None:
        finetune_config = FINETUNE_CONFIG
    if dir_name is None:
        dir_name = OUTPUT_DIR
    os.makedirs(dir_name, exist_ok=True)

    # 1. Split data
    censor_threshold = compute_threshold_from_split(
        rawdata.labels, censor_split, censor_region
    )
    if verbose:
        print(f'  censor_threshold = {censor_threshold:.4f}')

    train_subset, nontrain_subset = train_test_split(
        rawdata, test_size=2 * model_config.split, random_state=random_state
    )
    val_subset, test_subset = train_test_split(
        nontrain_subset, test_size=0.5, random_state=random_state
    )

    # 2. build noised train/val datasets & train on noised data
    if noise_type == 'ynoise':
        train_data = Ynoised_dataset(
            train_subset,
            mag_noise=noise_param,
            threshold=censor_threshold,
            targetregion=censor_region,
        )
        val_data = Ynoised_dataset(
            val_subset,
            mag_noise=noise_param,
            threshold=censor_threshold,
            targetregion=censor_region,
        )
    elif noise_type == 'xnoise':
        train_data = Xnoised_dataset(
            train_subset,
            scorerange=noise_param,
            threshold=censor_threshold,
            targetregion=censor_region,
            preset='medium',
        )
        val_data = Xnoised_dataset(
            val_subset,
            scorerange=noise_param,
            threshold=censor_threshold,
            targetregion=censor_region,
            preset='medium',
        )
    elif noise_type == 'omission':
        filtered_train = omit_sensitive_data(
            train_subset, censor_threshold, censor_region, omit_frac=noise_param
        )
        filtered_val = omit_sensitive_data(
            val_subset, censor_threshold, censor_region, omit_frac=noise_param
        )
        train_data = dgldataset(filtered_train)
        val_data = dgldataset(filtered_val)
    else:
        raise ValueError(f"Unknown noise_type '{noise_type}'. "
                         "Must be 'ynoise', 'xnoise', or 'omission'.")

    test_data = dgldataset(test_subset)   # always clean

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph1, _ = train_data[0]
    model = GCN(graph1.ndata['feat'].shape[1], model_config.hdim).to(device)

    if verbose:
        print(f'  [step 2] Training on noised data ({noise_type})...')
    gcn_train(model, model_config, train_data, val_data, verbose=verbose)

    # evaluate after noised training (pre-finetune baseline)
    ytest = test_data.labels
    ymin = min(ytest)
    ymax = max(ytest)
    yhat_noised, rmse_noised = evaluate(model, (test_data.graphs, ytest))
    lower_rmse_noised = local_loss(ytest, yhat_noised, ymin, censor_threshold)
    upper_rmse_noised = local_loss(ytest, yhat_noised, censor_threshold, ymax)
    corr_noised = spearmanr(ytest, yhat_noised)[0]
    lower_corr_noised = local_spearman(ytest, yhat_noised, censor_threshold, above=False)
    upper_corr_noised = local_spearman(ytest, yhat_noised, censor_threshold, above=True)

    # 3. build clean sensitive fine-tuning subset
    if censor_region == 'above':
        sensitive_mask = train_subset['labels'] > censor_threshold
    else:
        sensitive_mask = train_subset['labels'] < censor_threshold

    sensitive_clean = train_subset[sensitive_mask].copy()

    if clean_frac == 0.0 or len(sensitive_clean) == 0:
        # No clean data available — skip fine-tuning
        if verbose:
            print('  [step 3] clean_frac=0.0, skipping fine-tuning.')
        results = {
            'noised_rmse': float(rmse_noised),
            'noised_lower_rmse': float(lower_rmse_noised),
            'noised_upper_rmse': float(upper_rmse_noised),
            'noised_corr': float(corr_noised),
            'noised_lower_corr': float(lower_corr_noised),
            'noised_upper_corr': float(upper_corr_noised),
            'finetuned_rmse': None,
            'finetuned_lower_rmse': None,
            'finetuned_upper_rmse': None,
            'finetuned_corr': None,
            'finetuned_lower_corr': None,
            'finetuned_upper_corr': None,
            'clean_frac': clean_frac,
            'n_finetune_samples': 0,
        }
        _save_results(results, dir_name, jobname)
        return results

    n_finetune = max(1, int(len(sensitive_clean) * clean_frac))
    finetune_subset = sensitive_clean.sample(n=n_finetune, random_state=random_state)
    finetune_data = dgldataset(finetune_subset)

    if verbose:
        print(
            f'  [step 3] Fine-tuning on {n_finetune}/{len(sensitive_clean)} '
            f'clean sensitive samples (clean_frac={clean_frac})...'
        )

    # Use full clean val data as validation during fine-tuning
    val_clean_data = dgldataset(val_subset)

    # 4. retrain whole model on clean sensitive subset
    gcn_train(model, finetune_config, finetune_data, val_clean_data, verbose=verbose)


    # 5. evaluate after retraining on clean data
    yhat_finetuned, rmse_finetuned = evaluate(model, (test_data.graphs, ytest))
    lower_rmse_finetuned = local_loss(ytest, yhat_finetuned, ymin, censor_threshold)
    upper_rmse_finetuned = local_loss(ytest, yhat_finetuned, censor_threshold, ymax)
    corr_finetuned = spearmanr(ytest, yhat_finetuned)[0]
    lower_corr_finetuned = local_spearman(
        ytest, yhat_finetuned, censor_threshold, above=False
    )
    upper_corr_finetuned = local_spearman(
        ytest, yhat_finetuned, censor_threshold, above=True
    )

    results = {
        'noised_rmse': float(rmse_noised),
        'noised_lower_rmse': float(lower_rmse_noised),
        'noised_upper_rmse': float(upper_rmse_noised),
        'noised_corr': float(corr_noised),
        'noised_lower_corr': float(lower_corr_noised),
        'noised_upper_corr': float(upper_corr_noised),
        'finetuned_rmse': float(rmse_finetuned),
        'finetuned_lower_rmse': float(lower_rmse_finetuned),
        'finetuned_upper_rmse': float(upper_rmse_finetuned),
        'finetuned_corr': float(corr_finetuned),
        'finetuned_lower_corr': float(lower_corr_finetuned),
        'finetuned_upper_corr': float(upper_corr_finetuned),
        'clean_frac': clean_frac,
        'n_finetune_samples': n_finetune,
    }
    _save_results(results, dir_name, jobname)
    return results

def _save_results(results, dir_name, jobname):
    path = f'{dir_name}/adversarial_results_{jobname}.json'
    with open(path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f'  Results saved to {path}')


def run_adversarial_sweep(rawdata):
    """
    Sweeps over all combinations of noise_type x censor_split x clean_frac
    and saves results per configuration.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary = []

    for censor_split in CENSOR_SPLITS:
        for noise_type in NOISE_TYPES:

            # determine noise params to sweep for this noise_type
            if noise_type == 'ynoise':
                noise_params = YNOISE_LEVELS
                noise_param_labels = [str(p) for p in noise_params]
            elif noise_type == 'xnoise':
                noise_params = XNOISE_RANGES
                noise_param_labels = [f'{p[0]}-{p[1]}' for p in noise_params]
            elif noise_type == 'omission':
                noise_params = OMIT_FRACTIONS
                noise_param_labels = [f'{p:.1f}' for p in noise_params]

            for noise_param, noise_label in zip(noise_params, noise_param_labels):
                for clean_frac in CLEAN_FRACS:

                    trial_results = []
                    for trial in range(NUM_TRIALS):
                        trial_seed = ADVERSARIAL_SEED + trial
                        set_seeds(trial_seed)

                        jobname = (
                            f'{noise_type}_noiseparam{noise_label}'
                            f'_split{censor_split}'
                            f'_cleanfrac{clean_frac}'
                            f'_trial{trial}'
                        )
                        dir_name = (
                            f'{OUTPUT_DIR}/{noise_type}'
                            f'_split{censor_split}'
                            f'_noiseparam{noise_label}'
                        )

                        print(
                            f'\n\033[46m[adversarial] {noise_type} | '
                            f'split={censor_split} | '
                            f'noise={noise_label} | '
                            f'clean_frac={clean_frac} | '
                            f'trial={trial+1}/{NUM_TRIALS}\033[0m'
                        )

                        result = adversarial_retrain_wrapper(
                            noise_type=noise_type,
                            noise_param=noise_param,
                            censor_region=CENSOR_REGION,
                            censor_split=censor_split,
                            clean_frac=clean_frac,
                            jobname=jobname,
                            dir_name=dir_name,
                            random_state=trial_seed,
                            verbose=True,
                            rawdata=rawdata,
                        )
                        trial_results.append(result)

                    # aggregate across trials
                    agg = _aggregate_trials(
                        trial_results, noise_type, noise_label,
                        censor_split, clean_frac
                    )
                    summary.append(agg)

    # save full summary
    summary_path = f'{OUTPUT_DIR}/adversarial_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f'\nFull summary saved to {summary_path}')

    # also save as csv
    df = pd.DataFrame(summary)
    df.to_csv(f'{OUTPUT_DIR}/adversarial_summary.csv', index=False)
    print(f'Summary CSV saved to {OUTPUT_DIR}/adversarial_summary.csv')
    return df


def _aggregate_trials(trial_results, noise_type, noise_label, censor_split, clean_frac):
    """Average metrics across trials, skip None values (clean_frac=0 fine-tune cols)."""
    agg = {
        'noise_type': noise_type,
        'noise_param': noise_label,
        'censor_split': censor_split,
        'clean_frac': clean_frac,
    }
    metric_keys = [
        'noised_rmse', 'noised_lower_rmse', 'noised_upper_rmse',
        'noised_corr', 'noised_lower_corr', 'noised_upper_corr',
        'finetuned_rmse', 'finetuned_lower_rmse', 'finetuned_upper_rmse',
        'finetuned_corr', 'finetuned_lower_corr', 'finetuned_upper_corr',
        'n_finetune_samples',
    ]
    for key in metric_keys:
        vals = [r[key] for r in trial_results if r[key] is not None]
        if vals:
            agg[f'{key}_mean'] = float(np.mean(vals))
            agg[f'{key}_std'] = float(np.std(vals))
        else:
            agg[f'{key}_mean'] = None
            agg[f'{key}_std'] = None
    return agg


if __name__ == '__main__':
    import urllib.request
    urllib.request.urlretrieve(
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv",
        "./lipophilicity.csv",
    )
    lipodata = pd.read_csv("./lipophilicity.csv")
    rawdata = lipodata.rename(columns={'exp': 'labels'})
    run_adversarial_sweep(rawdata)