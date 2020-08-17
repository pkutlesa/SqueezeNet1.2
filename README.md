# SqueezeNet1.2

Abstract


Developing space-efficient convolutional neural
networks (CNNs) while maintaining the level of
accuracy achieved by state-of-the-art models on
image-classification tasks has become an emerging
field of research. In this work, we explore the
SqueezeNet model proposed by Forrest N. Iandola et
al., who demonstrated that the model achieves
AlexNet-level accuracy on the ILSVRC 2012 while
requiring 50x fewer parameters (Iandola et al.,
2016). First, we reproduce these results on the
CIFAR-10 image classification task. Next, we
perform ablation studies to determine the impact of
various SqueezeNet model components on achieved
accuracy and the number of learned parameters.
Finally, we explore potential enhancements to the
SqueezeNet model in an attempt to further improve
the model’s accuracy and space complexity. In this
paper, we share three key findings: (1) the
SqueezeNet model achieves AlexNet-level accuracy
on the CIFAR-10 dataset while learning
substantially fewer parameters than the AlexNet
model; (2) the Fire modules that are fundamental to
SqueezeNet substantially reduce the number of
parameters learned by the model; and (3) we propose
SqueezeNet 1.2, an improved version of the
SqueezeNet 1.1 model that achieves a reduction in
test error of 7.7 percentage points while learning
14.8 percent fewer parameters on the CIFAR-10
image classification task.

