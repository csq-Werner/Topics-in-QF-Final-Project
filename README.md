# Topics-in-QF-Final-Project
Hi, we are Anyi Liu (刘安易), Sitong Wei(魏斯桐) and Songqi Cao(曹淞琪).

This our final project of Topics in Quantitative Finance in Summer 2020, taught by Taiho Wang, a visiting professor in National School of Development Peking Universiity and professor in Baruch College, the City University of New York.

In this project, we reproduced the unsupervised deep learning approach to solving partial integro-differential equations reported in Weilong Fu & Ali Hirsa (2022) Quantitative Finance, 22:8, 1481-1494.

### Code Structure

```shell
|-- model
    |-- benchmark.py -- Implementation of benchmark model
    |-- option.py    -- Option calculation
    |-- pricer.py    -- Network structure
    |-- process.py   -- Benchmark model process
|-- data             -- Train and val dataset
|-- {MODEL}_Am_call  -- Pretrained network for different model
|-- main.py 		 --	Run the network training
|-- NSD_FinalProject_AnyiLiu_SitongWei_SongqiCao.ipynb
					 -- Methods and experiment results
```

