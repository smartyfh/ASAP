# ASAP-User Satisfaction Estimation
This is the Pytorch implementation of our work: **Modeling User Satisfaction Dynamics in Dialogue via Hawkes Process. Fanghua Ye, Zhiyuan Hu, Emine Yilmaz. ACL 2023.** [[paper](https://arxiv.org/abs/2305.12594)]

## Abstract
Dialogue systems have received increasing attention while automatically evaluating their performance remains challenging. User satisfaction estimation (USE) has been proposed as an alternative. It assumes that the performance of a dialogue system can be measured by user satisfaction and uses an estimator to simulate users. The effectiveness of USE depends heavily on the estimator. Existing estimators independently predict user satisfaction at each turn and ignore satisfaction dynamics across turns within a dialogue. In order to fully simulate users, it is crucial to take satisfaction dynamics into account. To fill this gap, we propose a new estimator ASAP (s**A**tisfaction e**S**timation via H**A**wkes **P**rocess) that treats user satisfaction across turns as an event sequence and employs a Hawkes process to effectively model the dynamics in this sequence. Experimental results on four benchmark dialogue datasets demonstrate that ASAP can substantially outperform state-of-the-art baseline estimators.

## Model Architecture

<p align="center">
  <img src="models/ASAP.png" width="62%" />
</p>

<p align="center">The model architecture of ASAP</p>
