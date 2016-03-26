# Function Observers
This is a MATLAB toolbox for code pertaining to function observers. 
The prototypical example of a functional observer is the 
kernel observer and controller paradigm, early versions of which appeared
as
* Hassan A. Kingravi, Harshal Maske, and Girish Chowdhary. 
  *[Kernel Observers: Systems-Theoretic Modeling and Inference of 
   Spatiotemporally Varying Processes
   (http://hassanakingravi.com/papers/2015/NIPS_2015.pdf)]*, 
   NIPS Nonparametric Methods for Large Scale Representation Learning (2015)
* Hassan A. Kingravi, Harshal Maske, and Girish Chowdhary. 
  *[A Systems-Theoretic Approach for Data-Driven Modeling and Control
    of Spatiotemporally Evolving Processes
   (http://hassanakingravi.com/papers/2015/CDC_2015.pdf)]*, 
   NIPS Nonparametric Methods for Large Scale Representation Learning (2015)
The primary idea behind these methods is the modeling and control of 
spatiotemporally varying processes (i.e. stochastic phenomena that vary
over space AND time). Practical applications of these types of methods 
include ocean temperature modeling and monitoring, control of diffusive 
processes in power plants, optimal decision-making in contested areas with a 
patrolling enemy, disease propagation in urban population centers, and so on. 
In all of these scenarios, there are some commonalities: 
1. *Modeling*: building a predictive model of the process in play: 
    ![Modeling][http://hassanakingravi.com/Images/papers/inference_example.png]
2. *Monitoring*: estimating the latent state of the process from 
    a set of measurements, gathered from *sensors*:
    ![Monitoring][http://hassanakingravi.com/Images/papers/inference_example.png]
3. *Control/Exploitation*: either affecting the future state of the process
    directly, or using the current state of the process to make a decision. 
    ![Control][http://hassanakingravi.com/Images/papers/inference_example.png]
The geostatistics community has done a great deal of work in the area. Our
contributions lie in utilizing ideas from reproducing kernel Hilbert space 
to make modeling easier and more efficient, and utilizing ideas from systems
theory to minimize the number of sensors, and the control effort required for
actuation. This toolbox will allow researchers to explore these ideas in their
own work. 

# Goals
The goal behind this toolbox is to make a self-contained version of
the code we used to write our papers, to ensure useability and 
reproducibility. We code all of our methods in MATLAB as opposed to 
Python since we imagine that the majority of people interested in such 
time-series analysis have access to MATLAB in an academic setting. The 
popularity of methods such as deep neural networks has also given Python 
a shot in the arm however, and that language is now heavily used in both
academia and industry. Therefore, we will also release a Python version
of this code in the near future. 

# Requirements and Setup Instructions



