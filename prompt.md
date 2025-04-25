notes:
- create an undo button with shortcut 'a' (no need to support multiple undo operations, one is fine)
- move the skip shortcut to 's'
- add a 1.5 second time limit to label the image, after which the labeller automatically skips to the next unlabelled image. the point of this is to ensure that the label we are capturing is the visceral, knee jerk reaction that we are trying to capture, not the result of a deliberate judgement that we may have talked ourselves into. include comments in the code for this feature to this effect. you may need to create a system for persisting which images have been skipped and a system for making sure skipped images are not included in the training set, do this in whatever hacky way you pleasen
- For images that have been skipped, either by going over the time limit or manually skipped, do not display those images again until we have run out of images that are unseen, that haven't been skipped.
- it'd be nice to have a `flag` button where I can mark images that contain problems with the dataset. implement `flag` with shorcut 'f'. exclude flagged images from the training set. flagging is different from skipping
- change the classification user interface to include punchabi Instead of having punchability be a binary choice of either punchable or not, I want to change my data labeling scheme to be VERY punchable, punchable, NOT punchable, VERY NOT punchable. And I would like the shortcuts for those to be H, J, K, and L, respectively. 
- propose a system for to monitor consistency: track intra-rater reliability (e.g., re-rate 10% of images). low agreement (κ < 0.7) flags fatigue.
- propose a system for determining optimal performance on this task, which might simply involve a rater rating the dev set once the data is all labelled.
