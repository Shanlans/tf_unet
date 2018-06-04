# -*- coding: utf-8 -*-

import numpy as np
    
from tf_unet import image_util
from tf_unet import unet
from tf_unet import util
import glob

search_path = "D:\\pythonworkspace\\tf_unet\\tf_unet\\demo\\IRholder\\ImageResize\\*.png"


data_provider = image_util.ImageDataProvider(search_path, data_suffix=".png", mask_suffix='_label.png')

net = unet.Unet(channels=data_provider.channels, n_class=data_provider.n_class, layers=4, features_root=32)

#trainer = unet.Trainer(net, optimizer="momentum", opt_kwargs=dict(momentum=0.2))
trainer = unet.Trainer(net,batch_size=8, optimizer="adam",opt_kwargs=dict(learning_rate=0.00001))
path = trainer.train(data_provider, "./unet_trained", training_iters=32, epochs=100,dropout=0.5, display_step=2,write_graph = False,restore=False)
#path = trainer.train(data_provider, "./unet_trained", training_iters=20, epochs=50, display_step=2)