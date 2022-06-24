coseg_chairs:
      python train.py
        -heterodir "${data_dir}/chairs"
        -load_Path "${save_path_of_pretrained_model}$/best_model_chair_hgt"
        -writerdir "${save_path_of_middle_data}/chairs"
        -losstype "focal"
        -n_hid 120
        -n_layers 4

coseg_vases:
      python train.py
        -heterodir "${data_dir}/vases"
        -load_Path "${save_path_of_pretrained_model}$/best_model_vase_hgt"
        -writerdir "${save_path_of_middle_data}/vases"
        -losstype "focal"
        -n_hid 120
        -n_layers 4

coseg_aliens:
      python train.py
        -heterodir "${data_dir}/aliens"
        -load_Path "${save_path_of_pretrained_model}$/best_model_alien_hgt"
        -writerdir "${save_path_of_middle_data}/aliens"
        -losstype "focal"
        -n_hid 120
        -n_layers 4


