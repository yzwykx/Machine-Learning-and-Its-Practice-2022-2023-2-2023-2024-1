# SVM的实现以及核函数探究

* data_generate_m.py <br>
本代码的目的是生成符合大作业要求的MNIST数据集，并保存在images_m_train.npy, labels_m_train.npy, images_m_test.npy, labels_m_test.npy.
运行本文件需要准备MNIST数据集，该数据集可以从网址 http://yann.lecun.com/exdb/mnist/ 上下载，下载完成后请添加到路径 ./mnist_data/ 中.

* data_generate_c.py <br>
本代码的目的是生成负荷大作业要求的CIFAR10数据集，并保存在images_c_train.npy, labels_c_train.npy, images_c_test.npy, labels_c_test.npy.
运行本文件需要准备CIFAR10数据集，该数据集可以从网址 https://www.cs.toronto.edu/~kriz/cifar.html 上下载，下载完成后请添加到路径 ./cifar-10-batches-py/ 中.

* SVM2.py <br>
本文件包含SVM类的实现以及使用SMO优化的代码.

* SVM.py <br>
本文件是本次大作业的主代码文件，包含数据集的导入，参数初始化，训练以及测试的相关代码.

* 最终运行需要的文件目录（数据集需要另外准备） <br>

```python
-- cifar-10-batches-py  -- data_batch_1
                        -- data_batch_2
                        -- data_batch_3
                        -- data_batch_4
                        -- data_batch_5
                        -- test_batch
                        -- readme.html
                        -- batches.meta

-- mnist_data           -- t10k-images-idx3-ubyte.gz
                        -- t10k-labels-idx1-ubyte.gz
                        -- train-images-idx3-ubyte.gz
                        -- train-labels-idx1-ubyte.gz
-- data_generate_m.py
-- data_generate_c.py
-- SVM.py
-- SVM2.py
-- README.md
```

配置好所需环境后，在终端内输入以下代码可以运行一个实例：<br>
```python
python data_generate_m.py
python data_generate_c.py
python SVM.py
```

