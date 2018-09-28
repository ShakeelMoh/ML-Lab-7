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
#include <cstdlib>

using namespace std;

float e = 2.71828182845904523536;

//Calculates sigmoid output
float sigmoid(float x){

   float output = 1/(1+(pow(e, -x)));
   return output;

}

//Calculate hidden neuron output
float hiddenNeuronOutput(float x1, float x2, float x3,float weight[3]){
   
   float output = 1;//add bias
   output += x1*weight[0];
   output += x2*weight[1];
   output += x3*weight[2];

   output = sigmoid(output);
   //cout << output << "\n" << endl;
   return output;
   
}

//Calculate final node outputs
float outputNodeOutput(float x1, float x2, float x3, float weight[3]){

   float output = 1;//add bias
   
   output += x1 * weight[0];
   
   output += x2 * weight[1];
   
   output += x3 * weight[2];

   output = sigmoid(output);
   //cout << output << endl;
   
   return output;
}

//Calculate error term for output node
float outputErrorTerm(float output, float target){

   float errorOutput = output * (1-output) * (target - output);
   //cout << "OUTPUT: " << output << " TARGET" << target << endl;
   //cout << errorOutput << endl;
   return errorOutput;
}

//Calculate hidden error term for output node
float hiddenErrorTerm(float output, float weight1, float out1Error){
   
   float errorOutput = output * (1-output) * (weight1 * out1Error);
   //cout << errorOutput << "\n" << endl;
   return errorOutput;
}

float randFloat(){
   float random = ((float) rand())/(float) RAND_MAX;
   float diff = 1-0.5;
   float r = random * diff;
   return 0.5 + r;
}

int main (int argc, char *argv[]) {

   cout << "Back propagation ANN Part 2\n" << endl;
   
   vector<float*> inputs = vector<float*>();
   //Inputs
   float inputs1[3] = {0,0,0};
   inputs.push_back(inputs1);
   float inputs2[3] = {0,0,1};
   inputs.push_back(inputs2);
   float inputs3[3] = {0,1,0};
   inputs.push_back(inputs3);
   float inputs4[3] = {0,1,1};
   inputs.push_back(inputs4);
   float inputs5[3] = {1,0,0};
   inputs.push_back(inputs5);
   float inputs6[3] = {1,0,1};
   inputs.push_back(inputs6);
   float inputs7[3] = {1,1,0};
   inputs.push_back(inputs7);
   float inputs8[3] = {1,1,1};
   inputs.push_back(inputs8);
   
   //Output targets
   vector<float> targets = {0,1,1,0,1,0,0,1};
   
   float targetOutput = 0;
   
   //Weights into hidden nodes
   float w1 = randFloat();
   
   float w2 = randFloat();
   float w3 = randFloat();
   float w4 = randFloat();
   float w5 = randFloat();
   float w6 = randFloat();
   float w7 = randFloat();
   float w8 = randFloat();
   float w9 = randFloat();
   
   float w10 = randFloat();
   float w11 = randFloat();
   float w12 = randFloat();

   float weightsH1[3] = {w1, w2, w3};
   float weightsH2[3] = {w4, w5, w6};
   float weightsH3[3] = {w7, w8, w9};
   
   //Weights into outputs
   float weightsO1[3] = {w10, w11, w12};
   
   //Hidden node outputs
   float H1Output = 0;
   float H2Output = 0;
   float H3Output = 0;
   
   //Output node outputs
   float OutputNode1 = 0;
   
   
   //Error terms for outputs
   float ErrorTerm1 = 1;//just init to 1 so loop goes
   
   //Error term for 3 hidden nodes
   float hiddenError1 = 0;
   float hiddenError2 = 0;
   float hiddenError3 = 0;
   
   //Learning rate
   float n = 1;
   int c = -1;
   int count = 1;
   cout << "ITERATION: " << count << endl;
   
   bool setComplete = false;
   int errors = 0;
   float mse = 0;
   do {
      
      c++;
      
      //cout << c << " IS C" << endl;
      if (c == 8){
         c = 0;
         count++;
         cout << "Number of Errors: " << errors << endl;
         mse /= 2;
         cout << "Mean Squared Error: " << mse << endl << endl;
         cout << "ITERATION: " << count << endl;
         
         //cout << "Reset errors" << endl;
         errors = 0;
      }
      //Calculate output for Hidden Node 1
      //cout << "Hidden neuron 1 output: " << endl;
      H1Output = hiddenNeuronOutput(inputs[c][0], inputs[c][1], inputs[c][2], weightsH1);
      
      //Calculate output for Hidden Node 2
      //cout << "Hidden neuron 2 output: " << endl;
      H2Output = hiddenNeuronOutput(inputs[c][0], inputs[c][1], inputs[c][2], weightsH2);
      
      //Calculate output for Hidden Node 3
      //cout << "Hidden neuron 3 output: " << endl;
      H3Output = hiddenNeuronOutput(inputs[c][0], inputs[c][1], inputs[c][2], weightsH3);
      
      //Calculate output for Output Node 1 and MSE
      //cout << "Training Example " << c << endl;
      //cout << "Ouput Node Output: " << endl;
      OutputNode1 = outputNodeOutput(H1Output, H2Output, H3Output, weightsO1);

      mse += pow(targets[c] - OutputNode1, 2);
      
      ErrorTerm1 = outputErrorTerm(OutputNode1, targets[c]);
      
      cout << "Output : "<< OutputNode1 << "\t|" << "Error for Output Node: " << ErrorTerm1 <<endl;
      //cout << ErrorTerm1 << endl;
      
      if (ErrorTerm1 > 0.001 || ErrorTerm1 < -0.001){
         errors++;
         //cout << "ERRORS ADDING: " << errors << endl;
      }
      //Calculate hidden node errors
      hiddenError1 = hiddenErrorTerm(H1Output, weightsO1[0], ErrorTerm1);
      hiddenError2 = hiddenErrorTerm(H2Output, weightsO1[1], ErrorTerm1);
      hiddenError3 = hiddenErrorTerm(H3Output, weightsO1[2], ErrorTerm1);
      
      //New weights for output layer
      weightsO1[0] += (n * ErrorTerm1 * H1Output);
      weightsO1[1] += (n * ErrorTerm1 * H2Output);
      weightsO1[2] += (n * ErrorTerm1 * H3Output);
      
      //New weights for hidden layer
      weightsH1[0] += (n * hiddenError1 * inputs[c][0]);
      weightsH1[1] += (n * hiddenError1 * inputs[c][1]);
      weightsH1[2] += (n * hiddenError1 * inputs[c][2]);
      
      weightsH2[0] += (n * hiddenError2 * inputs[c][0]);
      weightsH2[1] += (n * hiddenError2 * inputs[c][1]);
      weightsH2[2] += (n * hiddenError2 * inputs[c][2]);
      
      weightsH3[0] += (n * hiddenError3 * inputs[c][0]);
      weightsH3[1] += (n * hiddenError3 * inputs[c][1]);
      weightsH3[2] += (n * hiddenError3 * inputs[c][2]);
      //cout << setComplete << " SET COMPLETE" << endl;
      //cout << "//////" << errors << "////" << c << endl;
      
   } while ((errors != 0) || (c != 7));
   cout << "Iteration ran: " << count << " times" << endl;
   cout << "Number of Errors: " << errors << endl;
   cout << "Mean Squared Error: " << mse << endl << endl;
   return 0;

}
