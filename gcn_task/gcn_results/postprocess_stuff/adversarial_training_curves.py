import json
import os
import urllib.request

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.backends.backend_pdf import PdfPages

urllib.request.urlretrieve(
    'https://github.com/google/fonts/raw/main/ofl/ibmplexmono/IBMPlexMono-Regular.ttf',
    'IBMPlexMono-Regular.ttf',
)
fe = font_manager.FontEntry(fname='IBMPlexMono-Regular.ttf', name='plexmono')
font_manager.fontManager.ttflist.append(fe)
plt.rcParams.update({
    'axes.facecolor': '#f5f4e9',
    'grid.color': '#AAAAAA',
    'axes.edgecolor': '#333333',
    'figure.facecolor': '#FFFFFF',
    'axes.grid': False,
    'axes.prop_cycle': plt.cycler('color', plt.cm.Dark2.colors),
    'font.family': fe.name,
    'ytick.left': True,
    'xtick.bottom': True,
    'figure.dpi': 150,
})

RESULTS_BASE = '../../OUTPUTS/adversarial_results'
CENSOR_SPLITS = [0.1, 0.5, 0.9]
CLEAN_FRACS   = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
NUM_TRIALS    = 5

NOISE_CONFIGS = {
    'ynoise':   [str(p) for p in [0.0, 0.5, 2.0, 5.0]],
    'xnoise':   [f'{p[0]}-{p[1]}' for p in [[1.0, 1.0], [0.8, 1.0], [0.3, 0.4], [0, 0.05]]],
    'omission': [f'{p:.1f}' for p in [0.0, 0.5, 1.0]],
}


def _load(noise_type, noise_label, censor_split, clean_frac, trial):
    jobname = (
        f'{noise_type}_noiseparam{noise_label}'
        f'_split{censor_split}'
        f'_cleanfrac{clean_frac}'
        f'_trial{trial}'
    )
    dir_name = f'{RESULTS_BASE}/{noise_type}_split{censor_split}_noiseparam{noise_label}'
    fpath = f'{dir_name}/training_curves_{jobname}.json'
    if not os.path.exists(fpath):
        return None
    with open(fpath) as f:
        return json.load(f)


def _make_grid(noise_type, censor_split, clean_frac, phase):
    noise_labels = NOISE_CONFIGS[noise_type]
    ncols = len(noise_labels)
    fig, axs = plt.subplots(
        nrows=NUM_TRIALS, ncols=ncols,
        figsize=(ncols * 2.5 + 1, NUM_TRIALS * 2),
        dpi=150,
    )
    if ncols == 1:
        axs = axs.reshape(-1, 1)

    for trial in range(NUM_TRIALS):
        for j, noise_label in enumerate(noise_labels):
            ax = axs[trial, j]
            d = _load(noise_type, noise_label, censor_split, clean_frac, trial)

            if d is None:
                ax.text(0.5, 0.5, 'missing', ha='center', va='center',
                        transform=ax.transAxes, fontsize=7)
            elif phase == 'noised':
                ax.plot(d['noised_train_loss'], color='C0', lw=0.8, label='train')
                ax.plot(d['noised_val_loss'],   color='C1', lw=0.8, label='val')
            elif phase == 'finetuned':
                if d['ft_train_loss'] is None:
                    ax.text(0.5, 0.5, 'skipped', ha='center', va='center',
                            transform=ax.transAxes, fontsize=7)
                else:
                    ax.plot(d['ft_train_loss'], color='C0', lw=0.8, label='train')
                    ax.plot(d['ft_val_loss'],   color='C1', lw=0.8, label='val')

            if trial == 0:
                ax.set_title(noise_label, fontsize=8)
            if j == 0:
                ax.set_ylabel(f'Trial {trial + 1}', fontsize=8)
            ax.tick_params(labelsize=6)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper right', fontsize=7, ncol=2)

    phase_label = 'noised training' if phase == 'noised' else 'adversarial retraining'
    fig.suptitle(
        f'{phase_label} | {noise_type} | split={censor_split} | clean_frac={clean_frac}',
        fontsize=10,
    )
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    os.makedirs('training_curves', exist_ok=True)

    for clean_frac in CLEAN_FRACS:
        for phase, prefix in [('noised', 'noised'), ('finetuned', 'finetuned')]:
            if phase == 'finetuned' and clean_frac == 0.0:
                continue
            pdf_path = f'training_curves/{prefix}_cleanfrac{clean_frac}.pdf'
            with PdfPages(pdf_path) as pdf:
                for censor_split in CENSOR_SPLITS:
                    for noise_type in ['ynoise', 'xnoise', 'omission']:
                        fig = _make_grid(noise_type, censor_split, clean_frac, phase)
                        pdf.savefig(fig, bbox_inches='tight')
                        plt.close(fig)
            print(f'Saved {pdf_path}')
