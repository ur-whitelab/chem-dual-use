import papermill as pm
import os
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

