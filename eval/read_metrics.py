import os
import json
import xlwt

#####################################################  
# save_path_parent = "results/full"
# save_path_parent = "results/wo_lc"
# save_path_parent = "results/wo_poseopt"
# save_path_parent = "results/wo_sample"
# save_path_parent = "results/wo_depthprior"
# save_path_parent = "results/wo_poseprior"
#####################################################  


paths = [
    "results/full",
    "results/wo_lc",
    "results/wo_poseopt",
    "results/wo_sample",
    "results/wo_depthprior",
    "results/wo_poseprior",
]


dataset_types = [
    # 'replica',
    # 'tum',
    'waymo',
]

scenes = {
    'replica': [
        'office0',
        'office1',
        'office2',
        'office3',
        'office4',
        'room0',
        'room1',
        'room2',
    ],
    'tum': [
        '360',
        'desk',
        'desk2',
        'floor',
        'plant',
        'room',
        'rpy',
        'teddy',
        'xyz',
    ],
    'waymo': [
        '13476',
        '100613',
        '106762',
        '132384',
        '152706',
        '153495',
        '158686',
        '163453',
        '405841',
    ],
}

for save_path_parent in paths:
    all_metrics = {}
    
    for dataset_type in dataset_types:
        for scene in scenes[dataset_type]:
            metrics_file = os.path.join(
                save_path_parent,
                dataset_type,
                scene,
                'metrics.json'
            )

            # Prepare to collect metrics for all scenes
            # if 'all_metrics' not in locals():
            #     all_metrics = {}
            if dataset_type not in all_metrics:
                all_metrics[dataset_type] = {}

            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                all_metrics[dataset_type][scene] = {
                    'PSNR': metrics.get('PSNR', None),
                    'SSIM': metrics.get('SSIM', None),
                    'LPIPS': metrics.get('LPIPS', None),
                    'ATE': metrics.get('ATE', None),
                }
            else:
                print(f'{metrics_file} does not exist!!!')
                all_metrics[dataset_type][scene] = {
                    'PSNR': None,
                    'SSIM': None,
                    'LPIPS': None,
                    'ATE': None,
                }

    # After the loops, write to .xls
    wb = xlwt.Workbook()
    for dataset_type in all_metrics:
        ws = wb.add_sheet(dataset_type)
        
        # Write header: first cell empty, then scene names, then 'Avg'
        ws.write(0, 0, '')
        for j, scene in enumerate(scenes[dataset_type]):
            ws.write(0, j+1, scene)
        ws.write(0, len(scenes[dataset_type])+1, 'Avg')
        
        # Write metric names in first column
        metric_names = ['PSNR', 'SSIM', 'LPIPS', 'ATE']
        for i, metric in enumerate(metric_names):
            ws.write(i+1, 0, metric)
            row_values = []
            for j, scene in enumerate(scenes[dataset_type]):
                value = all_metrics[dataset_type].get(scene, {}).get(metric, '')
                # Format value as per instructions
                if value is not None and value != '':
                    try:
                        float_value = float(value)
                        row_values.append(float_value)
                        if metric == 'PSNR':
                            value_fmt = f"{float_value:.2f}"
                        else:
                            value_fmt = f"{float_value:.3f}"
                    except Exception:
                        value_fmt = value
                else:
                    value_fmt = ''
                ws.write(i+1, j+1, value_fmt)
                
            # Compute and write average in the last column
            avg_col = len(scenes[dataset_type]) + 1
            if row_values:
                avg_value = sum(row_values) / len(row_values)
                if metric == 'PSNR':
                    avg_fmt = f"{avg_value:.2f}"
                else:
                    avg_fmt = f"{avg_value:.3f}"
            else:
                avg_fmt = ''
            ws.write(i+1, avg_col, avg_fmt)
            
    wb_path = os.path.join(save_path_parent, 'all_metrics.xls')
    print(f'\nSaved work book as {wb_path}!\n')
    wb.save(wb_path)
