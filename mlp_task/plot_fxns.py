import matplotlib.pyplot as plt
import pandas as pd

# Right now the plotting functions can only plot up to 4 tasks
# may need to adjust it... Need to figure out how to set optimal figsize

def plot_trainingcurves(tasks, results, figname=""):
    all_losses = results['train_loss']
    all_val_losses = results['val_loss']

    fig, axs = plt.subplots(nrows=1, ncols=4, sharey=False, dpi=300 ,figsize=(12,3))
    axs = axs.flatten()
    for i, task in enumerate(tasks):
        if i > 3:
          print("Can only plot up to 4 tasks")
          break
        task_name = task[0]
        ax = axs[i]
        ax.plot(all_losses[task_name])
        ax.plot(all_val_losses[task_name])
        ax.set_title(task_name)
        ax.set_xlabel('epoch')
        ax.set_ylabel('RMSE')
    plt.tight_layout()
    if figname:
        plt.savefig(figname,dpi=300)


def plot_parityplots(tasks, results, threshold=0, figname=""):
    y_test = results['y_test']
    fig, axs = plt.subplots(nrows=1, ncols=4, sharey=False, dpi=300 ,figsize=(12,3))
    axs = axs.flatten()
    for i, task in enumerate(tasks):
        if i > 3:
          print("Can only plot up to 4 tasks")
          break
        task_name = task[0]
        ax = axs[i]
        preds = results['pred'][task_name]
        lower_error = results['lower_error'][task_name]
        upper_error = results['upper_error'][task_name]

        ax.scatter(y_test, preds, s=10, alpha=0.5)
        ax.plot([-10,10],[-10,10])
        ax.plot([threshold,threshold],[-10,10],'--',alpha=0.5)
        ax.set_xlim(-10,10)
        ax.set_ylim(-10,10)
        ax.text(-5,-9,f'rmse:\n{lower_error:.3f}', horizontalalignment='center')
        ax.text(5,7,f'rmse:\n{upper_error:.3f}', horizontalalignment='center')
        ax.set_xlabel('true labels')
        ax.set_ylabel('predicted labels')
        ax.set_title(task_name)
    plt.tight_layout()
    if figname:
        plt.savefig(figname,dpi=300)

def create_dataframe(full_results):
    # filter results then create DataFrame
    # TODO: this assumes the sensitive region is above the sensitivity threshold - will need to update the code
    data = {}
    data['x noise'] = full_results.get('x_noise_level')
    data['y noise'] = full_results.get('y_noise_level')
    data['omit?'] = full_results.get('omit')
    data['% omitted'] = full_results.get('omit_fraction')  # This will be None if 'omit_fraction' happens to not exist
    
    data['overall RMSE'] = full_results['overall_error_mean']
    data['overall RMSE std'] = full_results['overall_error_std']
    data['s=0 RMSE'] = full_results['lower_error_mean']
    data['s=0 RMSE std'] = full_results['lower_error_std']
    data['s=1 RMSE'] = full_results['upper_error_mean']
    data['s=1 RMSE std'] = full_results['upper_error_std']
    
    data['over_corr'] = full_results.get('overall_corr_mean')
    data['over_corr std'] = full_results.get('overall_corr_std')
    data['s=0 corr'] = full_results.get('lower_corr_mean')
    data['s=0 corr std'] = full_results.get('lower_corr_std')
    data['s=1 corr'] = full_results.get('upper_corr_mean')
    data['s=1 corr std'] = full_results.get('upper_corr_std')

    return pd.DataFrame.from_dict(data)