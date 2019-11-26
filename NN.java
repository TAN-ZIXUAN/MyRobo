package MyRobo;

import MyRobo.Interface.NeuralNetInterface;

import java.io.*;
import java.util.Random;

public class NN implements NeuralNetInterface {

    //Config (bias = 1, added in input layer and hidden layer:the last item)
    private static boolean ifBinary = true; //True: Binary representation 0,1    False: Bipolar representation -1,0,1
    private static double totalErrorThreshold =  0.05;  //Total error should less than this value

    private int argNumInputs;     //The number of inputs in the input vector
    private int argNumHidden;     //The number of nodes in this single hidden layer
    //private int argNumOutputs;   //The number of outputs in the input vector

    private double argLearningRate;     //The learning rate coefficient
    private double argMomentumRate;     //The momentum coefficient



    private double argA;                //Integer lower bound of sigmoid used by the output neuron only
    private double argB;                //Integer upper bound of sigmoid used by the output neuron only
    private double lowerBoundW;     //the bound of weights [-0.5,0.5]
    private double upperBoundW;


    //add bias
    //let the last item be bias ;
    private int numInputs = argNumInputs + 1;
    private int numHidden = argNumHidden + 1;
    //private int numOutputs = argNumOutputs;

    //arrays to store the neuron values in input, hidden and output layers
    private double[] inputLayers = new double[numInputs];   //contain bias
    double[] hiddenAfterActivation = new double[numHidden];


    //delta of neurons in hidden layer and output layer
    private double[] deltaHiddenLayer = new double[argNumHidden]; //bias node in hidden layer won't change and its value will always be 1, so no delta for hiddenLayer[4];
    //private double[] deltaOutputLayer = new double[numOutputs];

    //arrays to store weights from input layer to hidden
    private double[][] curtInput2HiddenWeights = new double[argNumHidden][numInputs]; //no input to hidden layers weights for bias node in hidden layer
    private double[][] deltaInput2HiddenWeights = new double[argNumHidden][numInputs];

    //arrays to store weights from hidden layer to output layer
    private double[] curtHidden2OutputWeights = new double[numHidden];
    private double[] deltaHidden2OutputWeights = new double[numHidden];

    //training set X & Y
    private double[][]trainSetX;
    private double[]trainSetY;



    public NN(int argNumInputs, int argNumHidden, double argLearningRate, double argMomentumRate, double argA, double argB, double lowerBoundW, double upperBoundW) {
        this.argNumInputs = argNumInputs;
        this.argNumHidden = argNumHidden;
        this.argLearningRate = argLearningRate;
        this.argMomentumRate = argMomentumRate;
        this.argA = argA;
        this.argB = argB;
        this.lowerBoundW = lowerBoundW;
        this.upperBoundW = upperBoundW;
        //initializeWeights();
    }

    @Override
    public double sigmoid(double x) {
        if (ifBinary) return 1 / (1 + Math.exp(-x)); //Return a binary sigmoid of the input X
        else return( 2 / (1 + Math.exp(-x)) - 1); //Return a bipolar sigmoid of the input X
    }

    //TODO try using ReLU as activation function

    @Override
    public double customSigmoid(double x) {
        //a general sigmoid with asymptotes bounded by (a,b)
        return (argB - argA) / (1 + Math.exp(-x)) - (-argA);
    }

    public double derivative_customSigmoid(double x) {
        double D;
        D = (1.0 / (argB - argA)) * (x - argA) * (argB - x);
        return D;
    }

    @Override
    public void initializeWeights() {

        //initialize weights from input layer to hidden
        //for bias: no weights from input layer to bias node in hidden layer
        /*for(int hidden = 0; hidden < argNumHidden; hidden++) {

            for(int input = 0; input < numInputs; input++) { //last item is the bias, no weights for bias from input to hidden layer
                curtInput2HiddenWeights[hidden][input] = getRandomDouble(lowerBoundW,upperBoundW);
                deltaInput2HiddenWeights[hidden][input] = 0.0;
            }
        }

        //initialize weights from hidden layer to output layer

        for(int hidden = 0;hidden < numHidden;hidden++) {
            curtHidden2OutputWeights[hidden] = getRandomDouble(lowerBoundW,upperBoundW);
            deltaHidden2OutputWeights[hidden] = 0.0;
        }*/

        for (double[] array:curtInput2HiddenWeights) {
            for (double element:array)
                element = getRandomDouble(lowerBoundW,upperBoundW);
        }
        for (double[] array:deltaInput2HiddenWeights) {
            for (double elment:array)
                elment = 0;
        }

        for (double element:curtHidden2OutputWeights) {
            element = getRandomDouble(lowerBoundW,upperBoundW);
        }
        for (double elment:deltaHidden2OutputWeights)
            elment = 0.0;





    }

    //method: get random double value in a range
    public double getRandomDouble(double lowerBound, double upperBound) {

        double random, result;
        random = new Random().nextDouble();
        result = lowerBound+(random*(upperBound-lowerBound));
        return result;

    }

    @Override
    public void zeroWeights() {

        for (double[] array : curtInput2HiddenWeights) {
            for (double element : array) {
                element = 0;
            }
        }

        for (double[] array : deltaInput2HiddenWeights) {
            for (double element : array) {
                element = 0;
            }
        }

        for (double elment: curtHidden2OutputWeights) {
            elment = 0;
        }

        for (double elment: deltaHidden2OutputWeights)
            elment = 0;
    }

    /**
     *
     * @param x: The input Vector. double array.
     * @return Value returned by th LUT or NN for input vector
     *
     * input X
     * output Y(after activation)
     */
    @Override
    public double outputFor(double[] x) {

        double[] hiddenLayers = new double[numHidden];  //contain bias
        double outputLayers;
        double[] hiddenAfterActivation = new double[numHidden];
        double outputAfterActivation;

        //calculate the sum of weights*input = value of hidden layer neurons
        for (int i = 0; i < argNumHidden; i++) {
            // for input bias
            hiddenLayers[i] = curtInput2HiddenWeights[i][argNumInputs]*1.0; //for input bias
            // for actual input
            for (int j = 0; j < argNumInputs; j++) {
                hiddenLayers[i] += curtInput2HiddenWeights[i][j]*x[j];
            }
            //hidden layer value after activation
            hiddenAfterActivation[i] = customSigmoid(hiddenLayers[i]);
        }

        //for bias in hidden layer (the last item)
        hiddenLayers[argNumHidden] = 1;
        hiddenAfterActivation[argNumHidden] = 1;

        //calculate the sum of weights*hidden = output
        outputLayers = 0;
        for (int i = 0; i < numHidden; i++)
            outputLayers += curtHidden2OutputWeights[i]*hiddenAfterActivation[i];

        //output after activation
        outputAfterActivation = customSigmoid(outputLayers);

        return outputAfterActivation;
    }



    //return squared error
    @Override
    public double train(double[] x, double argValue) { //x: input vector argValue: desired output
        double[] hiddenLayers = new double[numHidden];  //contain bias
        double outputLayers;

        double outputAfterActivation;

        //# Forward propagation: calculate s and y=f(s)
        //## get the output y = outputFor(x)
        outputAfterActivation = outputFor(x);
        //## calculate the derivative y' = y * (1 - y)
        double y_ = outputAfterActivation * (1 - outputAfterActivation);

        //# Backward propagation
        //## output error: delta_y = (c-y)*f'(y)
        double outputError = (argValue - outputAfterActivation) * y_;
        //## hidden layer error
        double[] hiddenLayerError = new double[numHidden];
        hiddenLayerError[argNumHidden] = 0;//bias will always be 1 so no error for bias
        for (int i = 0; i < argNumHidden; i++) {
            hiddenLayerError[i] = curtHidden2OutputWeights[i] * outputError * (hiddenAfterActivation[i] - argA) * (argB - hiddenAfterActivation[i]) / (argB - argA);
        }

        //# update weights
        //## update weights from hidden to output
        for (int i = 0; i < numHidden; i++) {
            double tempHidden2OutputWeights = curtHidden2OutputWeights[i];
            curtHidden2OutputWeights[i] += (argLearningRate * outputError * hiddenAfterActivation[i]) + deltaHidden2OutputWeights[i] * argMomentumRate;
            deltaHidden2OutputWeights[i] = curtHidden2OutputWeights[i] - tempHidden2OutputWeights;
        }
        //update weights of bias in hidden layers
        double tempHiddenBiasWeights = curtHidden2OutputWeights[argNumHidden];
        curtHidden2OutputWeights[argNumHidden] += (argLearningRate * outputError * 1) + deltaHidden2OutputWeights[argNumHidden]*argMomentumRate;
        deltaHidden2OutputWeights[argNumHidden] = curtHidden2OutputWeights[argNumHidden] - tempHiddenBiasWeights;

        //## update weights from input to hidden
        for (int i = 0; i < argNumHidden; i++) {
            for (int j = 0; j < argNumInputs; j++) {
                double tempInput2HiddenWeights = curtInput2HiddenWeights[i][j];
                curtInput2HiddenWeights[i][j] += (argLearningRate * hiddenLayerError[i] * x[j]) + deltaInput2HiddenWeights[i][j] * argMomentumRate;
                deltaInput2HiddenWeights[i][j] = curtInput2HiddenWeights[i][j] - tempInput2HiddenWeights;
            }
            //update bias in input layer
            double tempInputBiasWeights = curtInput2HiddenWeights[i][argNumInputs];
            curtInput2HiddenWeights[i][argNumInputs] += (argLearningRate * hiddenLayerError[i] * 1) + deltaInput2HiddenWeights[i][argNumInputs] * argMomentumRate;
            deltaInput2HiddenWeights[i][argNumInputs] = curtInput2HiddenWeights[i][argNumInputs] - tempInputBiasWeights;
        }
        //return squared error
        return Math.pow((outputAfterActivation - argValue), 2);
    }

    //save weights
    @Override
    public void save(File argFile) {
        PrintStream wStream = null;
        try {
            float weight;
            wStream = new PrintStream(new FileOutputStream(argFile));
            //weights: input to hidden
            for (int i = 0; i < argNumHidden; i++)
                for (int j = 0; j < numInputs; j++) {
                    weight = (float) curtInput2HiddenWeights[i][j];
                    wStream.println(String.format("%s", weight));
                }

            //weights: hidden to output
            for (int i = 0; i <  numHidden; i++) {
                weight = (float) curtHidden2OutputWeights[i];
                wStream.println(String.format("%s", weight));
            }

            if (wStream.checkError()) {
                System.err.println("Can not Save!");
            }
            wStream.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        finally {
            try { if (wStream != null) {
                wStream.close();
            }

            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    //load weights
    @Override
    public void load(File argFileName) throws IOException {
        BufferedReader rBuffer = null;
        String line;
        try{ initializeWeights();
            int count = 1;
            System.out.println("loading weights");
            rBuffer = new BufferedReader(new FileReader(argFileName));
            for (int i = 0; i < argNumHidden; i++)
                for (int j = 0; j <  numInputs; j++) {
                    line = rBuffer.readLine();
                    if (line != null) {
                        curtInput2HiddenWeights[i][j] = Double.parseDouble(line);
                    }
                }

            for (int i = 0; i < numHidden; i++) {
                line = rBuffer.readLine();
                if (line != null) {
                    curtHidden2OutputWeights[i] = Double.parseDouble(line);
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            try {
                if (rBuffer != null) {
                    rBuffer.close();
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

    }




}
