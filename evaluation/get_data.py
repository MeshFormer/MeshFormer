import gdown as gd 
import os 

if __name__=='__main__':
    dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(dir, 'data')
    alien_data_dir = os.path.join(data_dir, 'coseg_aliens')
    alien_data_dir_0 = os.path.join(alien_data_dir, '0')
    alien_data_dir_1 = os.path.join(alien_data_dir, '1')
    chair_data_dir = os.path.join(data_dir, 'coseg_chairs')
    chair_data_dir_0 = os.path.join(chair_data_dir, '0')
    chair_data_dir_1 = os.path.join(chair_data_dir, '1')
    vase_data_dir = os.path.join(data_dir, 'coseg_vases')
    vase_data_dir_0 = os.path.join(vase_data_dir, '0')
    vase_data_dir_1 = os.path.join(vase_data_dir, '1')


    
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
        os.mkdir(alien_data_dir)
        os.mkdir(alien_data_dir_0)
        os.mkdir(alien_data_dir_1)
        os.mkdir(chair_data_dir)
        os.mkdir(chair_data_dir_0)
        os.mkdir(chair_data_dir_1)
        os.mkdir(vase_data_dir)
        os.mkdir(vase_data_dir_0)
        os.mkdir(vase_data_dir_1)


        
    pretrain_dir = os.path.join(dir, 'pre_trained_model')
    alien_pre_dir = os.path.join(pretrain_dir, 'aliens')
    alien_pre_dir_data = os.path.join(alien_pre_dir, 'data')
    alien_pre_dir_pred = os.path.join(alien_pre_dir, 'pretrained')
    chair_pre_dir = os.path.join(pretrain_dir, 'chairs')
    chair_pre_dir_data = os.path.join(chair_pre_dir, 'data')
    chair_pre_dir_pred = os.path.join(chair_pre_dir, 'pretrained')
    vase_pre_dir = os.path.join(pretrain_dir, 'vases')
    vase_pre_dir_data = os.path.join(chair_pre_dir, 'data')
    vase_pre_dir_pred = os.path.join(chair_pre_dir, 'pretrained')



    if not os.path.exists(pretrain_dir):
        os.mkdir(pretrain_dir)
        os.mkdir(alien_pre_dir)
        os.mkdir(alien_pre_dir_data)
        os.mkdir(alien_pre_dir_pred)
        os.mkdir(chair_pre_dir)
        os.mkdir(chair_pre_dir_data)
        os.mkdir(chair_pre_dir_pred)
        os.mkdir(vase_pre_dir)
        os.mkdir(vase_pre_dir_data)
        os.mkdir(vase_pre_dir_pred)


    print('Download the alien data.')

    alien_data_url = "https://drive.google.com/uc?id=1lFAVO8zn0ey0VKZmJW6596k3GRIM-C-y"
    gd.download(alien_data_url, alien_pre_dir_data)
    print('End download data.')
    print('Download the alien pretrained model.')
    alien_pred_url = "https://drive.google.com/uc?id=1_xqd8E1b-KM42-58D_DX5Sc8k_RxSNZ5"
    gd.download(alien_pred_url, alien_pre_dir_pred)
    print('End download model.')
    print('You can evaluate alien dataset.\n')


    print('Download the chair data.')
    chair_data_url = "https://drive.google.com/uc?id=1qw2G_MG9V572Y5WVt-scjFECnjXLJQfy"
    gd.download(chair_data_url, chair_pre_dir_data)
    print('End download data.')
    print('Download the chair pretrained model.')
    chair_pred_url = "https://drive.google.com/uc?id=1jSnFNIIKeEa_XE-g515xrWTEMsaMCkul"
    gd.download(chair_pred_url, chair_pre_dir_pred)
    print('End download model.')
    print('You can evaluate chair dataset.\n')



    print('Download the vase data.')
    vase_data_url = "https://drive.google.com/uc?id=1kW-eAEXq6UXOrALvKSCx-XCo-y0AK2sR"
    gd.download(vase_data_url, vase_pre_dir_data)
    print('End download data.')
    print('Download the vase pretrained model.')
    vase_pred_url = "https://drive.google.com/uc?id=1foDcFhxxMgPaEBOxJPB_sx1hsSmomLNN"
    gd.download(vase_pred_url, vase_pre_dir_pred)
    print('End download model.')
    print('You can evaluate vase dataset.\n')
