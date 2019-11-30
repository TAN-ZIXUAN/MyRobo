package MyRobo;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class NNtestLynx {

    private static int argNumInputs = 2 ;
    private static int argNumHidden = 4;
    private static double argLearningRate = 0.2;
    private static double argMomentumRate = 0.9;
    private static int argA = -1;
    private static int argB = 1;
    private static double lowerBoundW = -0.5;
    private static double upperBoundW = 0.5;

    private static File rmsErrFile = new File("C:\\robocode\\robots\\MyRobo\\TanRobo.data\\Test_rmsErr.dat");
    private static List<Double> rmsErrArr = new ArrayList<Double>();

    public static void main(String[] args) throws IOException {

        File weights_file = new File("C:\\robocode\\robots\\MyRobo\\TanRobo.data\\Test_weights.dat");
        double[][] inputVector = {{1,-1,-1},{1,-1,1},{1,1,-1},{1,1,1}};
        double[] desiredOutput = {-1,1,1,-1};

        int count = 0;
        int convergeTime = 0;
        int iterationTime = 10;
        double[] totalErr;
        double rmsErr = 999999; //root-mean-square error

        for (int iter = 0; iter < iterationTime; iter++) {
            //NNLynx nn = new NNLynx(argNumInputs,argNumHidden,argLearningRate,argMomentumRate,argA,argB,true);
            NN_J nn = new NN_J(argNumInputs,argNumHidden,argLearningRate,argMomentumRate,argA,argB,lowerBoundW,upperBoundW);
            //NN nn = new NN(argNumInputs, argNumHidden, argLearningRate, argMomentumRate, argA, argB, lowerBoundW, upperBoundW);
            totalErr = new double[inputVector.length];


            int max_epochs = 10000;
            for (int i = 0; i < max_epochs; i++) {
                double sumErr = 0;
                for (int j = 0; j < inputVector.length; j++) {
                    totalErr[j]= Math.pow( nn.train(inputVector[j],desiredOutput[j]),2); //square error: error^2
                    //System.out.println(totalErr[j]);
                    sumErr += totalErr[j];

                }
                //System.out.println(sumErr);
                rmsErr = Math.sqrt(sumErr/inputVector.length); //rms error
                rmsErrArr.add(rmsErr);

                System.out.println(iter+": "+"epoch"+i + ":" + rmsErr);
                //nn.saveWeights(weights_file);
                printRMSErr(rmsErrFile);

            }

            if (rmsErr < 0.05) {
                convergeTime++;
                break; //stop training
            }
        }

    }

    public static void printRMSErr(File file) {
        PrintStream printFile = null;
        try {
            double myRMSErr;
            printFile = new PrintStream(new FileOutputStream(file));
            for(double elment:rmsErrArr) {
                printFile.println(elment);
            }

            if (printFile.checkError()) {
                System.out.println("Cannot save RMS err");
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        finally {
            try {
                {
                    if (printFile != null) {
                        printFile.close();
                    }
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }


    }
}
