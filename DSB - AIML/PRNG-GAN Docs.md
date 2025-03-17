## Pre-requisites
- **Pseudo-Random Number Generators (PRNGs)**: A PRNG is a deterministic algorithm that uses a secret internal state _S**i_ to process a random input seed _s_. The goal is to produce a large number sequence statistically indistinguishable from a truly random sequence.
- **Generative Adversarial Networks (GANs)**: GANs are a deep learning method. In this context, GANs are used to train a generator network to produce data that resembles a reference distribution. The generator minimizes the probability of a discriminator network accurately mapping the generator's outputs to a class label.
- **Discriminative and Predictive Approaches**: Two GAN architectures are used in the paper.
    - In the **discriminative approach**, the discriminator's inputs are number sequences drawn either from the generator or from a common source of randomness and labeled accordingly. The generator learns to mimic the distribution of random sequences to minimise the probability of correct classification by the discriminator.
    - In the **predictive approach**, each sequence of length _n_ produced by the generator is split. The first _n_ - 1 values are input to the predictor, and the _n_th value is the corresponding label. The predictor aims to maximise the probability of correctly predicting the _n_th value from the other values, while the generator minimises it.
- **Neural Network Architectures**:
    - **Generator**: The generator is a fully connected feed-forward (FCFF) neural network representing a function _G**O_(s, o1). It consists of FCFF layers with leaky ReLU and mod activations.
    - **Discriminator and Predictor**: The discriminator and the predictor are convolutional neural networks. The discriminator outputs a scalar representing the probability that the sequence is real. The predictor's architecture is the same as the discriminator's, but it takes a different input size and has a different output meaning.
- **Loss Functions and Optimizer**: Standard loss functions are used. In the discriminative case, both the generator and discriminator have least squares loss. In the predictive case, both have absolute difference loss. The Adam stochastic gradient descent optimizer is employed.
- **NIST Statistical Test Suite**: The NIST test suite is used to evaluate the randomness of the generated number sequences. The test suite consists of 188 distinct tests, each repeated 10 times, with 1,000,000 input bits consumed for each repetition. A test instance fails if its p-value is below a critical value (α = 0.01).
- **Experimental Procedure**: The experimental procedure involves initialising a dataset with a fixed random seed _s_ and varying offset values _o1i_. The GAN is trained, with the generator and the adversary performing gradient updates in turn. The generator's output is evaluated before and after training using the NIST test suite.

## Steps
To create your own PRNG model using GANs with Python and TensorFlow, follow these steps, drawing on the details provided in the source:

- **Set up the Environment:** Ensure you have Python and TensorFlow installed. You will likely need other common data science libraries like NumPy.
    
- **Design the Generator Model:**
    
    - The generator should be a fully connected feed-forward (FCFF) neural network.
    - Define the function _G**O_(s, o1) : B2 → B8, where 's' is a seed and 'o1' represents the PRNG state. Both _s_ and _o1_ are 16-bit unsigned integers. Therefore, the input is a vector consisting of a seed _s_ and a non-random scalar _o1_ representing the PRNG state.
    - Implement the generator with four hidden FCFF layers of 30 units each, and an output FCFF layer of 8 units.
    - Use leaky ReLU activation for the input and hidden layers.
    - Apply a 'mod' function as an activation in the output layer to map values into the desired range.
- **Design the Discriminator/Predictor Model:**
    
    - Implement either the discriminative or predictive approach.
    - Both the discriminator and the predictor are convolutional neural networks.
    - **Discriminator:** Implement the function *D(r) : B8 → *, where _r_ is a vector of length 8 produced by the generator or drawn from a standard source of pseudo-randomness. The discriminator outputs a scalar _p(true)_ in the range , representing the probability that the sequence belongs to either class. The discriminator consists of four stacked convolutional layers, each with 4 filters, kernel size 2, and stride 1, followed by a max pooling layer and two FCFF layers with 4 and 1 units, respectively.
    - **Predictor:** Implement the function _P (r**split) : B7 → B_, where _r**split_ is the generator’s output vector with the last element removed. The last element is used as the corresponding label for the predictor’s input. Apart from the input size and meaning of the output, the discriminator and the predictor share the same architecture.
- **Implement the GAN Framework:**
    
    - **Discriminative Approach:** The discriminator's inputs are number sequences drawn either from the generator or from a common source of randomness, and labeled accordingly. The generator learns to mimic the distribution of the random sequences to minimise the probability of correct classification.
    - **Predictive Approach:** Each sequence of length _n_ produced by the generator is split; the first _n_ − 1 values are the input to the predictor, and the _n_th value is the corresponding label. The predictor maximizes the probability of correctly predicting the _n_th value from the other values, while the generator minimizes it.
- **Define Loss Functions and Optimizer:**
    
    - Use standard loss functions.
    - **Discriminative Case:** The generator and discriminator both have least squares loss.
    - **Predictive Case:** The generator and the predictor both have absolute difference loss.
    - Use the Adam stochastic gradient descent optimizer.
- **Set up the Training Procedure:**
    
    - Initialise the evaluation dataset _Data_. It consists of input vectors _v**i_ ∈ B2 of the form _[s, o1i]_, such that the random seed _s_ in _v**i_ is fixed to the same arbitrary value for all _i_. The offset _o1i_ in _v**i_ starts at 0 for _v0_ and increments sequentially for the following vectors.
    - Train the networks, with the generator and the adversary performing gradient updates in turn.
- **Training Parameters:**
    
    - Train the GAN for 200,000 epochs over mini-batches of 2,048 samples.
    - The generator performs one gradient update per mini-batch and the adversary performs three.
    - Set the learning rate of the networks to 0.02.
    - The generator outputs floating-point numbers constrained to the range [0, 216−1], which are rounded to the nearest 16-bit integer for evaluation.
    - The evaluation dataset consists of 400 mini-batches of 2,048 input vectors each, for a total of 819,200 input samples.
    - The generator outputs 8 floating-point numbers for each input, each yielding 16 bits for the full output sequence. In total, each evaluation output thus consists of 104,857,600 bits, produced from a single random seed.
- **Evaluate with the NIST Statistical Test Suite:**
    
    - Apply the NIST test suite with default settings to assess the randomness of the generator's output.
    - The test suite consists of 188 distinct tests, each repeated 10 times, with 1,000,000 input bits consumed for each repetition.
    - A test instance fails if its p-value is below a critical value (α = 0.01).
- **Analyse and Refine:**
    
    - Measure the extent to which training the GANs improves the randomness properties of the generators by analyzing large quantities of outputs, produced for a single seed, using the NIST statistical test suite both before and after training.
    - Systematically approach model selection and hyper-parameter optimisation and investigate the learning process.

## Baby Steps
If you're new to implementing research papers, here's a breakdown of concrete baby steps to get you started with this PRNG-GAN model, focusing on making the process manageable and understandable.

- **Step 1: Understand the High-Level Goal**
    
    - The goal is to create a pseudo-random number generator (PRNG) using a generative adversarial network (GAN).
    - A PRNG is an algorithm that produces sequences of numbers that appear random.
    - A GAN is a machine learning model where two neural networks (a generator and a discriminator/predictor) compete to generate realistic data.
    - You'll be training a generator network to output pseudo-random sequences and using a discriminator/predictor network to help the generator improve.
- **Step 2: Set Up Your Development Environment**
    
    - **Install Python:** If you haven't already, download and install Python. A good starting point is to use Anaconda as it comes with most of the common data science packages such as numpy.
    - **Install TensorFlow:** Use pip to install TensorFlow, which you'll use for building and training your neural networks. Make sure that it is compatible with your hardware.
    - **Install Other Libraries:** You'll likely need NumPy for numerical operations. Install it using pip if it is not already installed.
- **Step 3: Implement the Generator**
    
    - **Define the Architecture:** The generator is a fully connected feed-forward (FCFF) neural network. It takes a seed and a PRNG state as input and outputs a sequence of numbers.
    - The generator function is defined as _G**O_(s, o1) : B2 → B8. The input is a vector with a seed _s_ and a non-random scalar _o1_. The seed and the scalar will be 16-bit unsigned integers. The output is a vector of length 8.
    - It consists of four hidden FCFF layers of 30 units each and an output FCFF layer of 8 units.
    - **Choose Activation Functions:** Use leaky ReLU for the hidden layers and a 'mod' function for the output layer.
    - **TensorFlow Code:** Start writing the TensorFlow code to create this network. Define the layers, activation functions, and input/output shapes.
- **Step 4: Implement the Discriminator or Predictor**
    
    - Choose whether you want to implement the discriminative or predictive approach. The predictive approach may give better results.
    - Both discriminator and predictor are convolutional neural networks.
    - **Discriminator:** The discriminator function is defined as *D(r) : B8 → [, where _r_ is a vector of length 8 produced by the generator. The output is the probability that the sequence is real. It consists of four convolutional layers (4 filters, kernel size 2, stride 1), a max pooling layer, and two FCFF layers.
    - **Predictor:** The predictor function is defined as _P (r**split) : B7 → B_, where _r**split_ is the generator’s output vector with the last element removed. It shares the same architecture as the discriminator, but it takes a different input size and has a different output meaning.
    - **TensorFlow Code:** Write the TensorFlow code for the chosen discriminator or predictor network.
- **Step 5: Define Loss Functions and Optimizer**
    
    - **Loss Functions:** Choose the appropriate loss functions based on your chosen approach.
        - Discriminative: Least squares loss for both generator and discriminator.
        - Predictive: Absolute difference loss for both generator and predictor.
    - **Optimizer:** Use the Adam optimiser.
    - **TensorFlow Code:** Implement the loss functions and set up the Adam optimiser for both the generator and discriminator/predictor.
- **Step 6: Create the Training Loop**
    
    - **Data Initialisation:** Create a dataset with a fixed random seed _s_ and varying offset values _o1i_.
    - **Training Parameters:** Set the training parameters such as epochs (200,000) and mini-batch size (2,048). Set the learning rate to 0.02.
    - **Gradient Updates:** Implement the training loop where the generator and discriminator/predictor update their weights in turn. The generator performs one gradient update per mini-batch and the adversary performs three.
    - **TensorFlow Code:** Write the code for the training loop, feeding data to the generator and discriminator/predictor, calculating losses, and applying the optimiser.
- **Step 7: Evaluation (NIST Test Suite)**
    
    - Download the NIST Statistical Test Suite.
    - Generate large quantities of output from the generator, both before and after training.
    - Run the NIST tests on the generated output to assess the randomness.
    - Analyse the results and compare the performance before and after training.
- **Step 8: Iterate and Refine**
    
    - Analyse the results of the NIST tests and identify areas for improvement.
    - Adjust the architecture of the generator or discriminator/predictor.
    - Fine-tune the training parameters, such as learning rate, batch size, and number of epochs.
    - Experiment with different activation functions or loss functions.

By breaking down the implementation into these baby steps, you can focus on understanding and implementing each component individually.