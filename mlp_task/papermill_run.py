import papermill as pm
import os
print(os.getcwd())

censor_region = 'above'
output_dir = 'OUTPUTS/notebooks'
os.makedirs(output_dir, exist_ok=True)
for censor_split in [0.1, 0.5, 0.9]:
    print(f'Running MLP notebooks for split {censor_split}')
    pm.execute_notebook(input_path='omit_mlp.ipynb', 
                        output_path=f'{output_dir}/omit_mlp_split{censor_split}_{censor_region}.ipynb',
                        allow_errors=True,
                        parameters={
                            'censor_split': censor_split,
                            'censor_region': censor_region,
                        },
    )
    pm.execute_notebook(input_path='xnoise_mlp.ipynb', 
                        output_path=f'{output_dir}/xnoise_mlp_split{censor_split}_{censor_region}.ipynb',
                        allow_errors=True,
                        parameters={
                            'censor_split': censor_split,
                            'censor_region': censor_region,
                        },
    )
    pm.execute_notebook(input_path='ynoise_mlp.ipynb', 
                        output_path=f'{output_dir}/ynoise_mlp_split{censor_split}_{censor_region}.ipynb',
                        allow_errors=True,
                        parameters={
                            'censor_split': censor_split,
                            'censor_region': censor_region,
                        },
    )
