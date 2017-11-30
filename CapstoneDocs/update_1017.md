# Capstone Update: Nov 10th, 2017

### Nutshell Version
Making progress but feeling less optimistic than a few days ago. Currently working on running expeirments with the artifical dataset named listops.

### Elaborate Version
I have results for a few variations of the model on SNLI which show that the chart-parsing (CP) model making hard decisions (ST-Gumbel softmax for decision making), and the same model making soft decisions (Gumbel sotmax) outperform a CP model that makes random choices during parsing. 
- They outperform the random model only slightly on the final task, NLI, with a 0.5% improvement. 
- However, they significantly outperform the random model on metrics measuring the rate at which the models correctly compose intermeiediate nodes lke noun phrases and preprositional phrases. 
- An important indicator of interesting, informtive trees is also the F1 score with ground truth tree, left-branching, and right-branching trees. Right-branching trees are more common in written English and in SNLI. We find that the CP model with ST-Gumbel decisions does have the highest F1 score with ground truth trees (0.38). However, for results from the same model the F1 score is higher with left-branching trees (0.39).

Overall, it's hard to say if the model is doing anything very informative just yet. I've printed and looked at tens of trees and haven't found anything too interpretable yet.

The current step is running models on an aritificial dataset and I've started work on this. This will allow for much shorter experimental run time and will allow me to quickly dia  few things,
- If a model that does superviised parsing doing sentence representation can solve this dataset, i.e. achieve high accuracy, but a standard RNN without parsing ability can not, is out CP model able to solve it too without the supervision?
- If so, then why do our trees with natural language look so strange?
- If not, what hill clinbing can we do? Does temeprature annealing help? Can we switch up the composition function to help?


