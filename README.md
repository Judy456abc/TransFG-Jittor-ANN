# TransFG-Jittor-ANN

这是一个使用 Jittor 框架复现 [TransFG](https://github.com/TACJu/TransFG) 模型的仓库。

## 工作简介
TransFG 提出了基于 Transformer 架构的细粒度视觉分类（FGVC）模型，旨在解决细粒度视觉分类任务中类间差异小、传统方法复杂且定位不准等问题，并在多个数据集取得 SOTA 性能。我们将 Pytorch 框架替换为 Jittor，并对比在五个不同的数据集（CUB-200-2011、Stanford Cars、Stanford Dogs、NABirds、iNaturalist 2017）上的性能。此外，我们对论文中提到的 split way, contrastive loss, value of margin $\alpha$ 等方法进行了消融实验，检验了这些方法的有效性。

## 安装依赖

1. **创建虚拟环境**：
   使用 Conda 创建一个指定版本的 Python 虚拟环境以隔离项目依赖，避免与系统中的其他项目冲突。

   ```bash
   conda create -n TransFG python=3.7.3
   ```

2. **激活虚拟环境**：
   在继续之前，激活你刚刚创建的虚拟环境。

   ```bash
   conda activate TransFG
   ```

3. **安装依赖**：
   使用 pip 安装 `requirements.txt` 文件中列出的依赖。

   ```bash
   pip3 install -r requirements.txt
   ```

4. **安装 Apex**：
   Apex 是一个用于混合精度和分布式训练的 PyTorch 扩展工具。我们需要从 GitHub 克隆 Apex 仓库，并安装它。

   ```bash
   git clone https://github.com/ptrblck/apex.git 
   cd apex
   git checkout apex_no_distributed
   pip install -v --no-cache-dir ./
   ```

请确保按照上述步骤顺序执行命令。如果在安装过程中遇到任何问题，可以参考 Apex 的 [官方文档](https://nvidia.github.io/apex) 或在 Issues 中寻求帮助。

## 数据集
> 我们在以下五个细粒度分类数据集上进行实验：

- [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/)
- [Stanford Cars](https://github.com/jhpohovey/StanfordCars-Dataset/tree/main/stanford_cars)
- [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/)
- [NABirds](https://dl.allaboutbirds.org/nabirds)
- [iNaturalist 2017](https://github.com/visipedia/inat_comp/tree/master/2017)

其中部分使用的数据集与原论文稍有出入（因原论文数据集 url 已失效，如 Stanford Cars）

## 训练
训练 TransFG-Jittor-ANN 在几个数据集上的基础指令如下：

```bash
CUDA_VISIBLE_DEVICES=0 python3 train.py --name [自定义] --dataset CUB_200_2011
```

注意，对于输入图像需要进行尺寸调整。除 iNat2017 数据集需设置为 304 $\times$ 304 (训练时采用随机裁剪，测试时采用中心裁剪) 外，其余数据集均统一默认为 448 $\times$ 448。
```bash
CUDA_VISIBLE_DEVICES=0 python3 train.py --name [自定义] --dataset INat2017 --img_size 304
```

## Citation
```
@article{he2021transfg,
  title={TransFG: A Transformer Architecture for Fine-grained Recognition},
  author={He, Ju and Chen, Jie-Neng and Liu, Shuai and Kortylewski, Adam and Yang, Cheng and Bai, Yutong and Wang, Changhu and Yuille, Alan},
  journal={arXiv preprint arXiv:2103.07976},
  year={2021}
}
```