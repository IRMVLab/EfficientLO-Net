### Environment Configuration：

torch > 1.5.0  

python > 3.6

##### We recommend using Docker to run the code：

```shell
docker pull ufoym/deepo:all-py36-cu102
```


```shell
cd pointnet2
python setup.py install && cd..
cd ops_pytorch/fused_conv_random_k && python setup.py install && cd../..
cd ops_pytorch/fused_conv_select_k && python setup.py install && cd../..
```

```shell
pip install yaml
pip install openpyxl
```


### Dataset Configuration：

```
├── data_root (data_root in config.py)
│   ├── 00 (KITTI Sequences)
│   ├── 01
│   ├── 02
│   ├── ..
```



### valuation Results and Log Files

```
├── experiment  (Automatically created in the project path after running)
│   ├── pwclonet_KITTI_2021-10-21_13-28 
│   │   ├── checkpoints
│   │   │   ├── pwclonet
│   │   ├── eval
│   │   │   ├── pwclonet_07
│   │   │   ├── pwclonet_..
│   │   ├── logs
│   │   │   ├── Backups, Logs, Evaluation Results
```

