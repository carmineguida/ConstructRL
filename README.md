# ConstructRL
An RL framework for the Construct 3 Game engine.

## Examples
The **examples** folder contains Construct 3 projects which can be imported into the Construct 3 editor: https://editor.construct.net

* **RLExampleDoor:** The simplest example of an agent going towards a door or avoiding lava.
* **RLExampleLunarLander:** The simplified classic "Lunar Lander" which uses continuous action values indead of discrete.
* **RLExampleMulti:** Multiple agents contributing to the training.
* **RLExampleImage:** Using an image (screenshot) as input rather than vectorized information.
* **RLExampleTrack:** An example of an image as well as vectorized data as input.


## Server
To run the server, simply execute the following on the command line (run the server first, then hit play in Construct):
```
python3 -m server
```

# Configuration (option)
In the **server directory** there is a file called **config.json** which can be used for various parameters:
* Port/Address for the server.
* Algorithm to use (PPO, SAC, etc.)
* Learning Rate, Gamma (-1 means "default")
* Activation function (blank means "default"), Neural Network sizes.


## Cite
If you are using this framework in research, please cite our paper using the following:
(bibtex information coming soon)