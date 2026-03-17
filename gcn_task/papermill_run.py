import papermill as pm
import os
import shutil
print(os.getcwd())

censor_region = 'above'
output_dir = 'OUTPUTS/notebooks'
os.makedirs(output_dir, exist_ok=True)
for censor_split in [0.1, 0.5, 0.9]:
    print(f'Running GCN notebooks for split {censor_split}')
    pm.execute_notebook(input_path='omission_gcn.ipynb',
                        output_path=f'{output_dir}/omission_gcn_split{censor_split}_{censor_region}.ipynb',
                        allow_errors=True,
                        parameters={
                            'censor_split': censor_split,
                            'censor_region': censor_region,
                        },
    )
    pm.execute_notebook(input_path='ynoise_gcn.ipynb',
                        output_path=f'{output_dir}/ynoise_gcn_split{censor_split}_{censor_region}.ipynb',
                        allow_errors=True,
                        parameters={
                            'censor_split': censor_split,
                            'censor_region': censor_region,
                        },
    )
    pm.execute_notebook(input_path='xnoise_gcn.ipynb',
                        output_path=f'{output_dir}/xnoise_gcn_split{censor_split}_{censor_region}.ipynb',
                        allow_errors=True,
                        parameters={
                            'censor_split': censor_split,
                            'censor_region': censor_region,
                        },
    )

postprocess_src = 'gcn_results/postprocess_stuff'
postprocess_dst = 'OUTPUTS/postprocess_stuff'
os.makedirs(postprocess_dst, exist_ok=True)
for fname in sorted(os.listdir(postprocess_src)):
    if fname.endswith('.ipynb'):
        shutil.copy(os.path.join(postprocess_src, fname), os.path.join(postprocess_dst, fname))
original_dir = os.getcwd()
os.chdir(postprocess_dst)
for fname in sorted(os.listdir('.')):
    if fname.endswith('.ipynb'):
        print(f'Running {fname}')
        pm.execute_notebook(input_path=fname,
                            output_path=fname,
                            allow_errors=True,
        )
os.chdir(original_dir)

