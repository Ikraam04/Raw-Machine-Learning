# Raw ML

## goals
The goal of this project was to essentially just create neural networks from scratch
I had a bunch of personal goals with this project:
1. Understand OOP in c++ - how classes work  and interact with eacother (asbtract classes, constructors, destructors) 
2. Understand smart pointers (because no one likes new and delete)
3. Understand the eigen library for linear algebra operations
4. implement the full neural network using custom classes (layer, tensor etc.)
5. test and train on Xor, MNIST, CIFAR-10 and more sets 
6. Later impelement the same thing using CUDA (on my GPU) and compare run times etc.


So far i've done 1-4.5, I have trained both xor and MNIST succesfully - now I am moving to CIFAR-10 and will probably have to implement convolutionl layers (conv2d, max pooling). 

After that I will move to CUDA and write my own kernels (literally not a clue what i am doing here)


#### REMINDER TO SELF CHANGE DEBUG MODE WHEN COMPILING
in `/build`
`rm -rf *`
`cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ..`
then remake
`ninja`
