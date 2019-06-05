from collections import OrderedDict
import torch

model = torch.load('../models/vgg_face_dag.pth')
new_dict = OrderedDict()
for k, v in model.items():
    if k == 'conv1_1.weight':
        new_dict['features.0.weight'] = model[k]
    elif k == 'conv1_1.bias':
        new_dict['features.0.bias'] = model[k]
    elif k == 'conv1_2.weight':
        new_dict['features.3.weight'] = model[k]
    elif k == 'conv1_2.bias':
        new_dict['features.3.bias'] = model[k]
    elif k == 'conv2_1.weight':
        new_dict['features.7.weight'] = model[k]
    elif k == 'conv2_1.bias':
        new_dict['features.7.bias'] = model[k]
    elif k == 'conv2_2.weight':
        new_dict['features.10.weight'] = model[k]
    elif k == 'conv2_2.bias':
        new_dict['features.10.bias'] = model[k]
    elif k == 'conv3_1.weight':
        new_dict['features.14.weight'] = model[k]
    elif k == 'conv3_1.bias':
        new_dict['features.14.bias'] = model[k]
    elif k == 'conv3_2.weight':
        new_dict['features.17.weight'] = model[k]
    elif k == 'conv3_2.bias':
        new_dict['features.17.bias'] = model[k]
    elif k == 'conv3_3.weight':
        new_dict['features.20.weight'] = model[k]
    elif k == 'conv3_3.bias':
        new_dict['features.20.bias'] = model[k]
    elif k == 'conv4_1.weight':
        new_dict['features.24.weight'] = model[k]
    elif k == 'conv4_1.bias':
        new_dict['features.24.bias'] = model[k]
    elif k == 'conv4_2.weight':
        new_dict['features.27.weight'] = model[k]
    elif k == 'conv4_2.bias':
        new_dict['features.27.bias'] = model[k]
    elif k == 'conv4_3.weight':
        new_dict['features.30.weight'] = model[k]
    elif k == 'conv4_3.bias':
        new_dict['features.30.bias'] = model[k]
    elif k == 'conv5_1.weight':
        new_dict['features.34.weight'] = model[k]
    elif k == 'conv5_1.bias':
        new_dict['features.34.bias'] = model[k]
    elif k == 'conv5_2.weight':
        new_dict['features.37.weight'] = model[k]
    elif k == 'conv5_2.bias':
        new_dict['features.37.bias'] = model[k]
    elif k == 'conv5_3.weight':
        new_dict['features.40.weight'] = model[k]
    elif k == 'conv5_3.bias':
        new_dict['features.40.bias'] = model[k]
    elif k == 'fc6.weight':
        new_dict['classifier.0.weight'] = model[k]
    elif k == 'fc6.bias':
        new_dict['classifier.0.bias'] = model[k]
    elif k == 'fc7.weight':
        new_dict['classifier.3.weight'] = model[k]
    elif k == 'fc7.bias':
        new_dict['classifier.3.bias'] = model[k]
    elif k == 'fc8.weight':
        new_dict['classifier.6.weight'] = model[k]
    elif k == 'fc8.bias':
        new_dict['classifier.6.bias'] = model[k]

torch.save(new_dict, '../models/vgg_face_dag_custom.pth')
