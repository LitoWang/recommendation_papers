# recommendation_papers
### *   means promissing
### [x] means validation is valid
## [一] 模型结构
### CTR/CVR 通用模型
- [x] [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
- [x] [Field-aware Factorization Machines for CTR Prediction](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf)
- [x] [Field-weighted Factorization Machines for Click-Through RatePrediction in Display Advertising](https://arxiv.org/pdf/1806.03514.pdf)
- [ ] [AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/pdf/1810.11921.pdf)
- [ ] [RaFM: Rank-Aware Factorization Machines](http://proceedings.mlr.press/v97/chen19n/chen19n.pdf)
- [ ] [FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction](https://www.groundai.com/project/fibinet-combining-feature-importance-and-bilinear-feature-interaction-for-click-through-rate-prediction/1)
- [ ] [CFM: Convolutional Factorization Machines for Context-Aware Recommendation](http://staff.ustc.edu.cn/~hexn/papers/ijcai19-cfm.pdf)
- [x] [Field-aware Neural Factorization Machine for Click-Through Rate Prediction](https://arxiv.org/pdf/1902.09096.pdf?forcedefault=true)
- [x] [Holographic Factorization Machines for Recommendation](https://www.researchgate.net/profile/Yi_Tay3/publication/330101551_Holographic_Factorization_Machines_for_Recommendation/links/5c2d6e1192851c22a3563aba/Holographic-Factorization-Machines-for-Recommendation.pdf) -> [__**note**__]()
- [ ] [Cross and Deep network for Ad Click Predictions](https://arxiv.org/pdf/1708.05123.pdf)
- [ ] [Deep Crossing: Web-Scale Modeling without Manually Crafted Combinatorial Features](https://www.kdd.org/kdd2016/papers/files/adf0975-shanA.pdf)
- [ ] [xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://arxiv.org/pdf/1803.05170.pdf)
- [ ] [Product-based Neural Networks for User Response Prediction over Multi-field Categorical Data](https://arxiv.org/pdf/1807.00311.pdf)
- [ ] [Product-based Neural Networks for User Response Prediction](https://arxiv.org/pdf/1611.00144.pdf)
- [ ] [DeepGBM: A Deep Learning Framework Distilled by GBDT for Online Prediction Tasks](http://delivery.acm.org/10.1145/3340000/3330858/p384-ke.pdf?ip=203.205.141.44&id=3330858&acc=OPENTOC&key=39FCDE838982416F%2E39FCDE838982416F%2E4D4702B0C3E38B35%2E9F04A3A78F7D3B8D&__acm__=1570593404_7bf8663e47961e69d446f0692f12001b)
- [ ] [InteractionNN: A Neural Network for Learning Hidden Features in Sparse Prediction](https://www.ijcai.org/proceedings/2019/0602.pdf)
- [ ] [High-order Factorization Machine Based on Cross Weights Network for Recommendation](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8840886)
- [ ] [Adaptive Factorization Network: Learning Adaptive-Order Feature Interactions](https://arxiv.org/pdf/1909.03276.pdf)
- [ ] [Quaternion Collaborative Filtering for Recommendation](https://arxiv.org/pdf/1906.02594.pdf)
- [ ] * [Fi-GNN: Modeling Feature Interactions via Graph Neural Networks for CTR Prediction](https://arxiv.org/pdf/1910.05552.pdf)
- [ ] [Exploring Content-based Video Relevance for Video Click-Through Rate Prediction](https://dl.acm.org/citation.cfm?id=3343031.3356053)

### 多任务
- [ ] [Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate](https://arxiv.org/pdf/1804.07931.pdf)
- [ ] [Predicting Different Types of Conversions with Multi-Task Learning in Online Advertising](https://arxiv.org/pdf/1907.10235.pdf)
- [ ] [Deep Bayesian Multi-Target Learning for Recommender Systems](https://arxiv.org/pdf/1902.09154.pdf) -> [__**note**__](https://git.code.oa.com/wechat_train/paper/blob/master/paper_notes_v3.docx)
- [ ] [A Causal Perspective to Unbiased Conversion Rate Estimation on Data Missing Not at Random](https://arxiv.org/pdf/1910.09337.pdf)

### 延迟反馈
- [ ] [Modeling Delayed Feedback in Display Advertising](http://olivier.chapelle.cc/pub/delayedConv.pdf)
- [ ] [A Nonparametric Delayed Feedback Model for Conversion Rate Prediction](https://arxiv.org/pdf/1802.00255.pdf)
- [ ] [A Practical Framework of Conversion Rate Prediction for Online Display Advertising](https://dl.acm.org/citation.cfm?id=2623634)
- [ ] * [Addressing Delayed Feedback for Continuous Training with Neural Networks in CTR prediction](https://arxiv.org/pdf/1907.06558.pdf)
- [ ] [Unbiased Learning to Rank with Unbiased Propensity Estimation](https://arxiv.org/pdf/1804.05938.pdf)
- [ ] * [Dual Learning Algorithm for Delayed Feedback in Display Advertising](https://arxiv.org/pdf/1910.01847.pdf)

## [二] 优化算法
### 累积后悔最小化
- [x] [Follow-the-Regularized-Leader and Mirror Descent: Equivalence Theorems and L1 Regularization](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/37013.pdf)
- [x] [Ad Click Prediction: a View from the Trenches](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/41159.pdf)
- [x] [Follow the Moving Leader in Deep Learning](http://proceedings.mlr.press/v70/zheng17a/zheng17a.pdf)

### 方差约减
- [ ] [A Stochastic Gradient Method with an Exponential Convergence Rate for Finite Training Sets](https://hal.inria.fr/file/index/docid/799158/filename/sag_arxiv.pdf)
- [x] [Accelerating Stochastic Gradient Descent using Predictive Variance Reduction](https://papers.nips.cc/paper/4937-accelerating-stochastic-gradient-descent-using-predictive-variance-reduction.pdf)
- [x] [Lookahead Optimizer: k steps forward, 1 step back](https://arxiv.org/pdf/1907.08610.pdf)

## [三] 特征构建
- [ ] [Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba](https://arxiv.org/pdf/1803.02349.pdf)
- [ ] [Real-time Personalization using Embeddings for Search Ranking at Airbnb](https://astro.temple.edu/~tua95067/kdd2018.pdf)
