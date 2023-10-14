from datetime import datetime
import os
import subprocess


def perform_run(params, run_name, log_dir="logs"):
    formatted_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
    run_name = f'{run_name}_{formatted_datetime}'
    params['output_dir'] = os.path.join(log_dir, run_name)
    if not os.path.exists(params['output_dir']):
        os.makedirs(params['output_dir'])
    log_file_path = os.path.join(params['output_dir'], 'log.txt')

    # Initialize the command list
    cmd = ['accelerate', 'launch', 'train_dreambooth_lora_sdxl.py']

    # Add parameters to the command list
    for key, value in params.items():
        if not isinstance(value, bool) or value: # if false we skip
            cmd.append(f'--{key}')
        if value is not None:
            if isinstance(value, list):
                for sub_value in value:
                    if isinstance(sub_value, str) and ' ' in sub_value: 
                        sub_value = f'"{sub_value}"'
                    cmd.append(sub_value)
            elif not isinstance(value, bool):  # Convert boolean to string
                if isinstance(value, str) and ' ' in value: 
                    value = f'"{value}"'
                cmd.append(value)

    # Open the log file for writing
    with open(log_file_path, 'w') as log_file:
        print(f'Running command:')
        print(' '.join(cmd))
        # Run the command and capture the output
        process = subprocess.Popen(cmd, stdout=log_file, stderr=log_file, text=True, bufsize=1)

        # Wait for the process to complete
        process.wait()


def main():
    train_image_dir = "images/train"
    validation_image_dir = "images/validation"

    validation_images = ['brad-pitt-thumbs-up.webp',
    'obama-thumbs-up.webp',
    'pexels-ketut-subiyanto-4909522.jpg']

    marker = '<thumbs_up>'

    validation_prompts = [f'a photo of Brad Pitt in a suit and sunglasses showing {marker} thumbs up', f'a photo of Barack Obama wearing a vest showing {marker} thumbs up', f'a photo of a black man at the beach showing {marker} thumbs up']
    
    for i, learning_rate in enumerate([1e-4, 1e-5, 1e-6, 1e-7]):

        # Define parameters in a dictionary
        params = {
            'pretrained_model_name_or_path': 'stabilityai/stable-diffusion-xl-base-1.0',
            'instance_data_dir': train_image_dir,
            'pretrained_vae_model_name_or_path': 'madebyollin/sdxl-vae-fp16-fix',
            'mixed_precision': 'fp16',
            'instance_prompt': f'a photo of a person showing {marker} thumbs up',
            'resolution': '1024',
            'train_batch_size': '1',
            'gradient_accumulation_steps': '4',
            'learning_rate': f'{learning_rate}',
            'report_to': 'wandb',
            'lr_scheduler': 'constant',
            'lr_warmup_steps': '0',
            'max_train_steps': '10', # 500
            'validation_prompt': f'A photo of Brad Pitt showing {marker} thumbs up',
            'validation_epochs': '25',
            'seed': '0',
            'push_to_hub': True,
            'center_crop': True,
            'validation_prompts': validation_prompts,
            'validation_image_dir': validation_image_dir,
            'validation_images': validation_images,
        }

        run_name = f'sweep_final_{i}'

        perform_run(params, run_name)


if __name__ == '__main__':
    # print current working directory
    print(os.getcwd())
    # list files in current working directory
    print(os.listdir())
    main()
