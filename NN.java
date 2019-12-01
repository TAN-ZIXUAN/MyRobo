package MyRobo;

import MyRobo.Interface.NeuralNetInterface;

import java.io.*;

public class NN implements NeuralNetInterface {

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
    private int numInputs;
    private int numHidden;

    //arrays to store the neuron values in hidden layers
    private double[] hiddenLayers;
    private double[] hiddenAfterActivation;

    //arrays to store weights from input layer to hidden
    public static double[][] curtInput2HiddenWeights;//no input to hidden layers weights for bias node in hidden layer
    public static double[][] deltaInput2HiddenWeights;

    //arrays to store weights from hidden layer to output layer
    public static double[] curtHidden2OutputWeights;
    public static double[] deltaHidden2OutputWeights;





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
        numInputs = argNumInputs + 1;
        numHidden = argNumHidden + 1;
        hiddenLayers = new double[numHidden];
        hiddenAfterActivation = new double[numHidden];
        curtInput2HiddenWeights = new double[argNumHidden][numInputs];
        deltaInput2HiddenWeights = new double[argNumHidden][numInputs];
        curtHidden2OutputWeights = new double[numHidden];
        deltaHidden2OutputWeights = new double[numHidden];
        zeroWeights();
        initializeWeights();

    }

    @Override
    public double sigmoid(double x) {
        return 0;
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
        for(int hidden = 0; hidden < argNumHidden; hidden++) {

            for(int input = 0; input < numInputs; input++) { //last item is the bias, no weights for bias from input to hidden layer
                curtInput2HiddenWeights[hidden][input] = getRandomDouble(lowerBoundW,upperBoundW);
                deltaInput2HiddenWeights[hidden][input] = 0.0;
            }
        }

        //initialize weights from hidden layer to output layer

        for(int hidden = 0;hidden < numHidden;hidden++) {
            curtHidden2OutputWeights[hidden] = getRandomDouble(lowerBoundW,upperBoundW);
            deltaHidden2OutputWeights[hidden] = 0.0;
        }

        /*for (double[] array:curtInput2HiddenWeights) {
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
            elment = 0.0;*/

    }

    //method: get random double value in a range
    public double getRandomDouble(double lowerBound, double upperBound) {

        return ((double) (Math.random()*(upperBound - lowerBound))) + lowerBound;
        /*double random, result;
        random = new Random().nextDouble();
        result = lowerBound+(random*(upperBound-lowerBound));
        return result;*/


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

         //contain bias
        double outputLayers;

        double outputAfterActivation;

        //calculate the sum of weights*input = value of hidden layer neurons
        for (int i = 0; i < argNumHidden; i++) {
            // for input bias
            hiddenLayers[i] = curtInput2HiddenWeights[i][argNumInputs]*1.0; //for input bias
            // for actual input
            for (int j = 0; j < argNumInputs; j++) {
                hiddenLayers[i] += curtInput2HiddenWeights[i][j] * x[j];
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
        //outputAfterActivation = customSigmoid(outputLayers);

        //leaky relu
        outputAfterActivation = Leaky_ReLu(outputLayers);

        return outputAfterActivation;
    }



    //return squared error
    @Override
    public double train(double[] x, double argValue) { //x: input vector argValue: desired output

        double outputAfterActivation;

        //# Forward propagation: calculate s and y=f(s)
        //## get the output y = outputFor(x)
        outputAfterActivation = outputFor(x);
        //## calculate the derivative y' = y * (1 - y)
        //double y_ = derivative_customSigmoid(outputAfterActivation);
        //relu
        double y_ = Leaky_ReLu_deri(outputAfterActivation);

        //# Backward propagation
        //## output error: delta_y = (c-y)*f'(y)
        double outputError = (argValue - outputAfterActivation) * y_;

        //# update weights
        //## update weights from hidden to output
        for (int i = 0; i < numHidden; i++) {
            deltaHidden2OutputWeights[i] = (argLearningRate * outputError * hiddenAfterActivation[i]) + deltaHidden2OutputWeights[i] * argMomentumRate;
            curtHidden2OutputWeights[i] += deltaHidden2OutputWeights[i];
        }
        //##update weights of bias in hidden layers
        deltaHidden2OutputWeights[argNumHidden] = (argLearningRate * outputError * 1) + deltaHidden2OutputWeights[argNumHidden]*argMomentumRate;
        curtHidden2OutputWeights[argNumHidden] += deltaHidden2OutputWeights[argNumHidden];



        //## hidden layer error
        double[] hiddenLayerError = new double[numHidden];
        hiddenLayerError[argNumHidden] = 0;//bias will always be 1 so no error for bias
        for (int i = 0; i < argNumHidden; i++) {
            //hiddenLayerError[i] = curtHidden2OutputWeights[i] * outputError * derivative_customSigmoid(hiddenAfterActivation[i]);
            //relu
            hiddenLayerError[i] = curtHidden2OutputWeights[i] * outputError * Leaky_ReLu_deri(hiddenAfterActivation[i]);
        }

        //## update weights from input to hidden
        for (int i = 0; i < argNumHidden; i++) {
            for (int j = 0; j < argNumInputs; j++) {
                deltaInput2HiddenWeights[i][j] = (argLearningRate * hiddenLayerError[i] * x[j]) + deltaInput2HiddenWeights[i][j] * argMomentumRate;
                curtInput2HiddenWeights[i][j] += deltaInput2HiddenWeights[i][j];
            }
            //update bias in input layer
            deltaInput2HiddenWeights[i][argNumInputs] = (argLearningRate * hiddenLayerError[i] * 1) + deltaInput2HiddenWeights[i][argNumInputs] * argMomentumRate;
            curtInput2HiddenWeights[i][argNumInputs] += deltaInput2HiddenWeights[i][argNumInputs];

        }
        //return squared error
        return Math.pow((outputAfterActivation - argValue), 2);
    }

    //save weights
    @Override
    public void save(File argFile) {
        PrintStream saveFile = null;
        try {
            float weight;
            saveFile = new PrintStream(new FileOutputStream(argFile));
            //weights: input to hidden
            for (int i = 0; i < argNumHidden; i++)
                for (int j = 0; j < numInputs; j++) {
                    weight = (float) curtInput2HiddenWeights[i][j];
                    saveFile.println(String.format("%s", weight));
                }

            //weights: hidden to output
            for (int i = 0; i <  numHidden; i++) {
                weight = (float) curtHidden2OutputWeights[i];
                saveFile.println(String.format("%s", weight));
            }

            if (saveFile.checkError()) {
                System.err.println("Can not Save!");
            }
            saveFile.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        finally {
            try { if (saveFile != null) {
                saveFile.close();
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

    public double Leaky_ReLu(double x){
        return Math.max(0.01*x,x);
    }

    public static double Leaky_ReLu_deri(double x){
        if(x>0){return 1;}
        else {return 0.01;}
    }



}
