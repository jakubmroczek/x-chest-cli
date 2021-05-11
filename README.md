# Simple tool to test our net, by with it manually

The script takes the state_dict from net.pth, loads it and predicts images.

Currently, we support squeeznet1_0, which is the one stored in net.pth. It was trained before, in one of our experiments.

Call the script with

`python main.py --image=./test`

If you want to test new images, you have to pyt them in the proper catalogue of `test`. This should be improved in the future.