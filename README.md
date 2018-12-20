# MSTGCN
Multi-Component Spatial-Temporal Graph Convolutional Networks (MSTGCN)

# References
AAAI 2019 《Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting》  
软件学报 2019 第三期 《面向交通流量预测的多组件时空图卷积网络》

# Datasets
We validate our model on two highway traffic datasets PeMSD4 and PeMSD8 from California. The datasets are collected by the Caltrans Performance Measurement System ([PeMS](http://pems.dot.ca.gov/)) ([Chen et al., 2001](https://trrjournalonline.trb.org/doi/10.3141/1748-12)) in real time every 30 seconds. The traffic data are aggregated into every 5-minute interval from the raw data. The system has more than 39,000 detectors deployed on the highway in the major metropolitan areas in California. Geographic information about the sensor stations are recorded in the datasets. There are three kinds of traffic measurements considered in our experiments, including total flow, average speed, and average occupancy.