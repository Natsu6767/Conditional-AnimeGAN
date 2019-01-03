# Conditional AnimeGAN
PyTorch implementation of conditional Generative Adversarial Network (cGAN) for Anime face generation conditioned on eye color and hair color.
<p align="center">
<img src="images/Generated_Anime_Faces.gif" title="Generated Data Animation" alt="Generated Data Animation">
</p>

**Row-1:** *Brown* Eyes *Blonde* Hair<br />
**Row-2:** *Blue* Eyes *Blue* Hair<br />
**Row-3:** *Red* Eyes *Green* Hair<br />
**Row-4:** *Purple* Eyes *Orange* Hair<br />
**Row-5:** *Green* Eyes *Purple* Hair<br />
**Row-6:** *Aqua* Eyes *Pink* Hair

You can download the dataset from the following [repo](https://github.com/m516825/Conditional-GAN).

## Training
Download the data and place it in the data/ directory. *(Optional) Run **`prepro.py`** to clean and preprocess the data.* Run **`train.py`** to start training. To change the hyperparameters of the network, update the values in the `param` dictionary in `train.py`.
Checkpoints will be saved by default in the `checkpoint` directory every 2 epochs.
By deafult, GPU will be used for training if available. *(Training on CPU is not recommended)*

**Loss Curve**
<p align="center">
<img src="images/Loss_Curve.png" title="Training Loss Curves" alt="Training Loss Curves">
</p>
<i>D: Discriminator, G: Generator</i>

## Generating New Images
To generate new images run **`generate.py`**.
```sh
python3 generate.py -load_path /path/to/pth/checkpoint -num_output n -eye_color c1 -hair_color c2
```
- Possible colors for eyes
```
['yellow', 'gray', 'blue', 'brown', 'red', 'green', 'purple', 'orange',
 'black', 'aqua', 'pink', 'bicolored']
```
- Possible colors for hair
```
['gray', 'blue', 'brown', 'red', 'blonde', 'green', 'purple', 'orange',
 'black', 'aqua', 'pink', 'white']
```
## Results
<table align='center'>
<tr align='center'>
<td> Training Data </td>
<td> cDCGAN after 50 epochs </td>
</tr>
<tr>
<td><img src = 'images/Training_Images.png'>
<td><img src = 'images/Epoch_50.png'>
</tr>
</table>

### Some Generated Samples:
**Blue Eyes Blonde Hair<br />
<img src="images/Blue-Blonde.png" height="128" width="170" title="Blue Eyes Blonde Hair" alt="Blue Eyes Blonde Hair"><br />
Red Eyes Blonde Hair<br />
<img src="images/Red-Blonde.png" height="128" width="170" title="Red Eyes Blonde Hair" alt="Red Eyes Blonde Hair"><br />
Green Eyes Purple Hair<br />
<img src="images/Green-Purple.png" height="128" width="170" title="Green Eyes Purple Hair" alt="Green Eyes Purple Hair"><br />
Red Eyes Green Hair<br />
<img src="images/Red-Green.png" height="128" width="170" title="Red Eyes Green Hair" alt="Red Eyes Green Hair"><br />
Aqua Eyes Pink Hair<br />
<img src="images/Aqua-Pink.png" height="128" width="170" title="Aqua Eyes Pink Hair" alt="Aqua Eyes Pink Hair"><br />
Red Eyes Purple Hair<br />**
<img src="images/Red-Purple.png" height="128" width="170" title="Red Eyes Purple Hair" alt="Red Eyes Purple Hair">

## References
1. **Mehdi Mirza, Simon Osindero.** *Conditional Generative Adversarial Nets.* [[arxiv](https://arxiv.org/abs/1411.1784)]
2. **Scott Reed, Zeynep Akata, Xinchen Yan, Lajanugen Logeswaran, Bernt Schiele, Honglak Lee.** *Generative Adversarial Text to Image Synthesis.* [[arxiv](https://arxiv.org/abs/1411.1784)]
3. **m516825/Conditional-GAN** [[repo](https://github.com/m516825/Conditional-GAN)]
4. **soumith/ganhacks** [[repo](https://github.com/soumith/ganhacks)]
