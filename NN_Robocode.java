package MyRobo;

import java.io.File;
import java.io.FileNotFoundException;
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

    private static int argNumInputs = numStates + numActions;
    private static int numTrainingVector = (3 * 2 * 8 * 6) * numActions; //each line of lut
    private static int argNumHidden = 14;
    private static double argLearningRate = 0.05;
    private static double argMomentumRate = 0.9;
    private static double argA = 0;
    private static double argB = 1;
    private static double lowerBoundW = -0.5;
    private static double upperBoundW = 0.5;
    private static NN nn = new NN(argNumInputs, argNumHidden, argLearningRate, argMomentumRate, argA, argB, lowerBoundW, upperBoundW);


    //Training NN with LUT
    public static void my(String[] args) throws FileNotFoundException {
        TanRobo robo = new TanRobo();
        LUT lut_NN = new LUT();
        File weights_file = new File("./TanRobo.data.weights.txt");
        robo.loadData();

        List<double[]> inputs = new ArrayList<double[]>();
        List<Double> output = new ArrayList<Double>();

        //initialize input vector
        for (int a = 0; a < SegDistance2target; a++) {
            for (int b = 0; b < SegGunHeat; b++)
                for (int c = 0; c < SegX; c++)
                    for (int d = 0; d < SegY; d++)
                        for (int action = 0; action < numActions; action++) {
                            double[] newInput = {
                                    2.0 * (double) a / (double) (SegDistance2target - 1) - 1,
                                    2.0 * (double) b / (double) (SegGunHeat - 1) - 1,
                                    2.0 * (double) c / (double) (SegX - 1) - 1,
                                    2.0 * (double) d / (double) (SegY - 1) - 1
                            };

                            double[] action_NN = new double[numActions];
                            Arrays.fill(action_NN, -1.0);
                            action_NN[action] = 1.0;

                            //combine state and action
                            newInput = DoubleStream.concat(Arrays.stream(newInput), Arrays.stream(action_NN)).toArray();

                            //todo confusing here
                            int crtState = States.Mapping[a][b][c][d];
                            double newOutput = (lut_NN.qTable[crtState][action] - -10) / 20;
                            if (lut_NN.qTable[crtState][action] != 0) {
                                inputs.add(newInput);
                                output.add((lut_NN.qTable[crtState][action] - -10) / 20);
                            }


                        }
        }

        int count = 0;
        int convergeTime = 0;
        int iterationTime = 10;
        double totalErr = 0;
        double sqrtErr = 999999; //root-mean-square error

        //Test Constants
        double inputVector[][] = new double[inputs.size()][argNumInputs];
        double outputVector[] = new double[inputs.size()];
        inputVector = inputs.toArray(inputVector);

        for (int i = 0; i < output.size(); i++) {
            outputVector[i] = output.get(i);
        }

        for (int k = 0; k < iterationTime; k++) {

            int max_epochs = 10000;
            for (int i = 0; i < max_epochs; i++) {
                totalErr = 0;
                for (int j = 0; j < inputVector.length; j++) {
                    totalErr += nn.train(inputVector[j], outputVector[j]);

                }
                sqrtErr = Math.sqrt(totalErr);

                System.out.println(i + ";" + sqrtErr);
                nn.save(weights_file);
            }

            if (sqrtErr < 0.05) {
                convergeTime++;
                break; //stop training
            }
        }
    }
}







