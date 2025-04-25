# ships-log.md
This document exists to Log my experiences and capture the decisions that I'm making over time based on what I experience as I work on this project.

# observations

started with resnet18, frozen except last layer (512 to 2 classes, ~1026 trainable params). celeba dataset, 200k images total, but small labeled subset (200 dev).
first curves: training acc 0.7759, dev acc 0.6702, dev loss 0.6259—overfitting hard. added weight decay (0.0005), new curves: training acc 0.7567, dev acc 0.6869, dev loss 0.6094. better, but still overfitting. dev acc noisy, loss dropping without acc gains (model’s overconfident on wrong preds).
switched to 4 classes (very punchable to very not punchable) for more signal. still overfitting—small labeled set’s the issue.
celeba has ~10k identities, ~20 images each. labeling per-image, not per-person, keeps nuance but risks inconsistency (same person, different punchability by angle).
for portfolio flex, added vit-tiny (5.7m params) to compare with resnet18 (11.7m params). vit-tiny needs more data but can generalize better with attention.
data’s the bottleneck: 200 dev images too small, 5k-10k training + 1k-2k dev is the goal. labeling 10k takes 3 hrs (1 sec/image)—20k max in 2 days, brutal.
ssl (masked autoencoders) can use all 200k celeba images without labels. pretrain vit-tiny, fine-tune on your 10k labeled. nvidia 3090 can handle it (~20-30 hrs, 100 epochs, batch 64).
inference plan: for ceos, average punchability over multiple images (5-10 per ceo) to reduce noise.
extras: grad-cam for resnet, attention maps for vit to show what model “sees” (eyes, smirks?). multi-modal models (vlms) could check your labeling consistency later.

stray thoughts:
- is "punchability" a linear combination of attractiveness, trustworthiness, dominance? using only ATD features, can we predict punchability, or is it a fourth distinct thing? if it is predictable, do we need all three? (ablative analysis)

# your actions
label 10k-20k celeba images (per-image, 4 classes). 2-day grind if needed—20k max.
develop some idea of what the highest possible level of performance is -- how do you perform re-rating your own images?
train resnet18 (frozen, last layer fine-tuned) on labeled set as baseline.
train vit-tiny (froze, last layer fine-tuned) to see how much transformers help
pretrain vit-tiny with mae on all 200k celeba images (ssl, ~20-30 hrs on 3090), fine-tune on your labeled set to see how much masked autoencoders help
explore multi-modal models images and text to see how they do at the task un-trained, 
compare resnet18 vs vit-tiny: dev acc, per-class f1, grad-cam/attention maps.
writeup: note overfitting, small data limits, and solo labeling bias. suggest more data, inter-rater reliability as future work.
