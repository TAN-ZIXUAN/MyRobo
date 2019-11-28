package MyRobo;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.DoubleStream;

//for training NN with LUT
public class NN_Robocode {

    private LUT lutForTraining = new LUT();
    public static final int SegDistance2target = 3;    //3 segmentation close:d<=200, medium:200<d<=400, far:d>400
    public static final int SegX = 8;
    public static final int SegY = 6;
    public static final int SegGunHeat = 2;

    private static int numStates = States.numStates;
    private static int numActions = Actions.numActions;

    private static int argNumInputs = 10 ;
    private static int argNumHidden = 14;
    private static double argLearningRate = 0.002;
    private static double argMomentumRate = 0.9;
    private static double argA = -2;
    private static double argB = 2;
    private static double lowerBoundW = -0.5;
    private static double upperBoundW = 0.5;
    //private static NN nn = new NN(argNumInputs, argNumHidden, argLearningRate, argMomentumRate, argA, argB, lowerBoundW, upperBoundW);
    private static File lutFile = new File ("C:\\robocode\\robots\\MyRobo\\TanRobo.data\\LUT.dat") ;


    //Training NN with LUT
    public static void main(String[] args) throws IOException {
        TanRobo robo = new TanRobo();
        LUT lut_NN = new LUT();
        File weights_file = new File("C:\\robocode\\robots\\MyRobo\\TanRobo.data\\weights.txt");
        lut_NN.load(lutFile);

        List<double[]> inputs = new ArrayList<double[]>();
        List<Double> output = new ArrayList<Double>();

        //initialize input vector
        //the desired output is the q value we load from LUT
        /**
         * read lut file first and normalize each value to [-1,1]
         * normalization: [a,b]
         * newData = a + (data - min)*(b - a)/(max -min)
         *
         */
        double[] _distance = new double[SegDistance2target];
        double[] _gunHeat = new double[SegGunHeat];
        double[] _x = new double[SegX];
        for (int a = 0; a < SegDistance2target; a++) {

            if (a == 0) {
                _distance[0]  = 200;
            }
            else if (a == 1 ) {
                _distance[1] = 400;
            }
            else {
                _distance[2] = 1000;
            }
            for (int b = 0; b < SegGunHeat; b++) {

                if (b == 0) {
                    _gunHeat[0] = 0;
                }
                else {
                    _gunHeat[1] = 1;
                }
                for (int c = 0; c < SegX; c++) {

                    _x[c] = (c + 1)*100;

                    for (int d = 0; d < SegY; d++) {
                        double[] _y = new double[SegY];
                        _y[d] = (d + 1)*100;

                        for (int action = 0; action < numActions; action ++) {
                            double[] newInput = {
                                    2.0 * _distance[a]/1000.0 - 1,
                                    2.0 * _gunHeat[b] /1.0 - 1,
                                    2.0 * _x[c] / 800.0 - 1,
                                    2.0 * _y[d] / 800.0 - 1
                            };

                            double[] action_NN = new double[numActions];
                            Arrays.fill(action_NN, -1.0);
                            action_NN[action] = 1.0;

                            //combine state and action
                            newInput = DoubleStream.concat(Arrays.stream(newInput), Arrays.stream(action_NN)).toArray();

                            int crtState = States.Mapping[a][b][c][d];

                            //for desired output we need to normalize it
                            double newOutput = (lut_NN.qTable[crtState][action]) / 2500;
                            if (lut_NN.qTable[crtState][action] != 0) {
                                inputs.add(newInput);
                                output.add((lut_NN.qTable[crtState][action]) / 2500);
                            }
                        }
                        }

                    }
                }


            }




        int count = 0;
        int convergeTime = 0;
        int iterationTime = 10;
        double[] totalErr;
        double rmsErr = 999999; //root-mean-square error

        //Test Constants
        double inputVector[][] = new double[inputs.size()][argNumInputs];
        double outputVector[] = new double[inputs.size()];
        inputVector = inputs.toArray(inputVector);

        for (int i = 0; i < output.size(); i++) {
            outputVector[i] = output.get(i);
        }

        for (int k = 0; k < iterationTime; k++) {
            NN nn = new NN(argNumInputs, argNumHidden, argLearningRate, argMomentumRate, argA, argB, lowerBoundW, upperBoundW);
            nn.initializeWeights();
            nn.load(weights_file);
            totalErr = new double[inputVector.length];



            int max_epochs = 10000;
            for (int i = 0; i < max_epochs; i++) {
                double sumErr = 0;
                for (int j = 0; j < inputVector.length; j++) {
                    totalErr[j]= nn.train(inputVector[j], outputVector[j]); //square error: error^2
                    sumErr += totalErr[j];

                }
                rmsErr = Math.sqrt(sumErr/inputVector.length); //rms error

                System.out.println("k: "+"epoch"+i + ":" + rmsErr);
                nn.save(weights_file);
            }

            if (rmsErr < 0.05) {
                convergeTime++;
                break; //stop training
            }
        }
    }

    public void LoadLUT() {
        BufferedReader reader = null;

    }
}







