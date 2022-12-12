# SJTU_DL_FIN_PROJECT

#### 1. 模块介绍：
>[color=#66ccff] **Rxy_Final_Job**
> **utils.py：** 封装各种自定义函数，例如*delaunay_triangulation*
> **data_generate.py：** 用于生成数据的脚本，可以根据命令行输入的参数下载对应数据。
> **data_loader.py：** 数据集加载模块，目前只支持二图匹配。
> **train.py** 训练模型的脚本，可以根据命令行输入调节训练超参数。
> **eval.py** 测试模型的脚本，可以根据命令行输入调节测试参数。
&ensp;&ensp;值得一提的是，目前我的搭建的训练测试框架可以支持不同的数据集，只需要使用的时候在命令行末尾加上- -*type='expected_data_type'(你想要的数据集)*，而且适配不同的模型（目前只支持三种）。想要训练什么模型只需要在使用时在命令行末尾加上- -*mode='expected_model'（你想要的模型）*。而且我的框架调参非常方便，只需要在训练的时候在命令行输入你想要的参数即可，例如：
```bash
python train.py --batch_size=4 --lr=0.001 --obj_resize=(256,256) --epochs=300 
--mode='pca_gm' --type='WillowObject' --lr_scheduling=True --classes='Face'

python eval.py --vggmodel='./model/vgg.pkl' --gmnetmodel='./model/gmnet.pkl' 
--obj_resize=(256,256) --mode='pca_gm' --type='WillowObject'  --classes='Face'
```
