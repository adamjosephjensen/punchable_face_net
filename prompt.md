goal: make a python script that makes a list of CelebA files we want to use for a training set.

Attribute file: /Users/adamjensen/Documents/celebA/CelebA/anno/list_attr_celeba.txt
Format: 1 for true, -1 for false
Target column: male
Target value: 1
head of attribute file:
202599
5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young
000001.jpg -1  1  1 -1 -1 -1 -1 -1 -1 -1 -1  1 -1 -1 -1 -1 -1 -1  1  1 -1  1 -1 -1  1 -1 -1  1 -1 -1 -1  1  1 -1  1 -1  1 -1 -1  1
000002.jpg -1 -1 -1  1 -1 -1 -1  1 -1 -1 -1  1 -1 -1 -1 -1 -1 -1 -1  1 -1  1 -1 -1  1 -1 -1 -1 -1 -1 -1  1 -1 -1 -1 -1 -1 -1 -1  1


images location: /Users/adamjensen/Documents/celebA/CelebA/Img/img_align_celeb
Example files:  067521.jpg, 101288.jpg, 135055.jpg

destination location: /Users/adamjensen/Documents/punchable_face_net/data/training_imgs.txt
format: one filename per line

requirements: given a number, produce a file that lists the filenames we want to train on. This should be deterministic, i.e. given "1000" it should put the same 1000 files every time in the file. All of the selected images should be of males, which we can learn from the attribute file.

concerns: up to this point, we have been copying files into a local folder for hosting. I think this is actually kind of dumb, because I already have the files, and like, why duplicate the data? I think the concern in the past was oh if we have a lot of files that could slow down the performance but I don't buy that. Opening a specific file in a directory should take the same ammount of time no matter how many files are in that directory, right? PLEASE CONFIRM.

related edits: need to edit the app to serve images from the images location. other changes may be necessary, think about this for a second, will the data labels still be correct?
