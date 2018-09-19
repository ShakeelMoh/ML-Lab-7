//ML Lab 7
//Back propagation 2

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <cmath>
#include <algorithm>

using namespace std;

float e = 2.71828182845904523536;

//Calculates sigmoid output
float sigmoid(float x){

   float output = 1/(1+(pow(e, -x)));
   return output;

}

//Calculate hidden neuron output
float hiddenNeuronOutput(float input[3], float weight[3]){
   
   float output = 0;
   for (int i = 0; i < 3; ++i){
      output += input[i]*weight[i];
      
   }
   output = sigmoid(output);
   cout << output << "\n" << endl;
   return output;
   
}

//Calculate final node outputs
float outputNodeOutput(float x1, float x2, float x3, float weight[3]){

   float output = 0;
   
   output += x1 * weight[0];
   
   output += x2 * weight[1];
   
   output += x3 * weight[2];

   output = sigmoid(output);
   cout << output << "\n" << endl;
   
   return output;
}

//Calculate error term for output node
float outputErrorTerm(float output, float target){

   float errorOutput = output * (1-output) * (target - output);
   cout << errorOutput << "\n" << endl;
   return errorOutput;
}

//Calculate hidden error term for output node
float hiddenErrorTerm(float output, float weight1, float out1Error){
   
   float errorOutput = output * (1-output) * (weight1 * out1Error);
   //cout << errorOutput << "\n" << endl;
   return errorOutput;
}

int main (int argc, char *argv[]) {

   cout << "Back propagation ANN Part 2\n" << endl;
   
   //Inputs
   float inputs[3] = {0,0,0};
   
   //Output targets
   float targetOutput = 0;
   
   //Weights into hidden nodes
   float weightsH1[3] = {1, 1, 1};
   float weightsH2[3] = {1, 1, 1};
   float weightsH3[3] = {1, 1, 1};
   
   //Weights into outputs
   float weightsO1[3] = {1, 1, 1};
   
   //Hidden node outputs
   float H1Output = 0;
   float H2Output = 0;
   float H3Output = 0;
   
   //Output node outputs
   float OutputNode1 = 0;
   
   //Output targets
   float targetOutput1 = 0;
   
   //Error terms for outputs
   float ErrorTerm1 = 0;
   
   //Error term for 3 hidden nodes
   float hiddenError1 = 0;
   float hiddenError2 = 0;
   float hiddenError3 = 0;
   
   //Learning rate
   float n = 0.1;

   //Calculate output for Hidden Node 1
   cout << "Hidden neuron 1 output: " << endl;
   H1Output = hiddenNeuronOutput(inputs, weightsH1);
   
   //Calculate output for Hidden Node 2
   cout << "Hidden neuron 2 output: " << endl;
   H2Output = hiddenNeuronOutput(inputs, weightsH2);
   
   //Calculate output for Hidden Node 3
   cout << "Hidden neuron 3 output: " << endl;
   H3Output = hiddenNeuronOutput(inputs, weightsH2);
   
   //Calculate output for Output Node 1 and MSE
   cout << "Ouput Node Output: " << endl;
   OutputNode1 = outputNodeOutput(H1Output, H2Output, H3Output, weightsO1);
   cout << "Error for Output Node 1: " << endl;
   ErrorTerm1 = outputErrorTerm(OutputNode1, targetOutput);
   
   
   //Calculate hidden node errors
   hiddenError1 = hiddenErrorTerm(H1Output, weightsO1[0], ErrorTerm1);
   hiddenError2 = hiddenErrorTerm(H2Output, weightsO1[1], ErrorTerm1);
   hiddenError3 = hiddenErrorTerm(H2Output, weightsO1[2], ErrorTerm1);
   
   //New weights for output layer
   weightsO1[0] += (n * ErrorTerm1 * H1Output);
   weightsO1[1] += (n * ErrorTerm1 * H2Output);
   weightsO1[2] += (n * ErrorTerm1 * H3Output);
   
   //New weights for hidden layer
   weightsH1[0] += (n * hiddenError1 * inputs[0]);
   weightsH1[1] += (n * hiddenError1 * inputs[1]);
   weightsH1[2] += (n * hiddenError1 * inputs[2]);
   
   weightsH2[0] += (n * hiddenError1 * inputs[0]);
   weightsH2[1] += (n * hiddenError1 * inputs[1]);
   weightsH2[2] += (n * hiddenError1 * inputs[2]);
   
   weightsH3[0] += (n * hiddenError1 * inputs[0]);
   weightsH3[1] += (n * hiddenError1 * inputs[1]);
   weightsH3[2] += (n * hiddenError1 * inputs[2]);
   
   return 0;

}
