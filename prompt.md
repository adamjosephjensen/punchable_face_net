notes:
- having an undo button would be nice
- it'd be nice to have a `flag` button where I can mark images that contain problems with the dataset. some of the men are labelled as women. <musing>In general: I wonder how many public data sets contain errors like this and what acceptable error rates are.</musing>
- I am noticing that this subjective task is difficult. How punchable should they be in order to be marked as punchable? how to hold a consistent standard? is it possible to prompt a multi-modal model for what it thinks of as punchable?
- in general I think I may actually have to scrape images of ceos and label them all by hand. hope not but that's what it feels like.
- it seems like a lot of these images have weird artifacts where the pixels are "smeared" up or down 
- important: i'm worried the web app might be serving the same image twice even after i have classified it. 
- uh oh, the labeller served 1002/1000 images. it should stop when I have classified all the images.
- I likely need to read about the data labelling techniques used in the paper on measuring trustworthiness of faces.
