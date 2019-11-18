package MyRobo;

import MyRobo.Interface.LUTInterface;
import robocode.RobocodeFileOutputStream;

import java.io.*;

public class LUT implements LUTInterface {


        /*int argNumInputs;
        int[] argVariableFloor;
        int[] argVariableCelling;*/
/*
energy: low middle high (energy of my own tank)
distance:close near far (distance between my tank and enemy)
actions:circle retreat, advance, fire
 */


    //2 states:energy, distance; 1 actions
    private int bestAction = 0;
    public double[][] qTable;


    public LUT() {
/*        this.argNumInputs = argNumInputs;
        this.argVariableFloor = argVariableFloor;
        this.argVariableCelling = argVariableCelling;*/

        qTable = new double[States.numStates][Actions.numActions];
        initialiseLUT();
    }

    @Override
    public void initialiseLUT() { //set all values in q_table to 0
        for(int i=0; i<States.numStates; i++)
            for(int j=0; j<Actions.numActions; j++) {
                qTable[i][j] = (Math.random() - 0.5) * 5000;
            }
    }

    @Override
    public int indexFor(double[] x) {
        return 0;
    }

    @Override
    public double outputFor(double[] x) {
        return 0;
    }

    @Override
    public double train(double[] x, double argValue) {
        return 0;
    }

    @Override
    public void save(File argFile) {

        PrintStream saveFile = null;
        try {
            saveFile = new PrintStream(new RobocodeFileOutputStream(argFile));
            for (int i = 0; i < States.numStates; i++)
                for (int j = 0; j < Actions.numActions; j++)
                    saveFile.println(new Double(qTable[i][j]));

            if (saveFile.checkError())
                System.out.println("Could not save the data!");

            saveFile.close();
        } catch (IOException e) {
            e.printStackTrace();
            System.out.println("IOException trying to write: " + e);
        }catch(NullPointerException e) {
            e.printStackTrace();
        }

        finally {
            try {
                if (saveFile != null)
                    saveFile.close();
            } catch (Exception e) {
                e.printStackTrace();
                System.out.println("Exception trying to close writer: " + e);
            }

        }


    }

    /**
     * Loads the LUT or neural net weights from file. The load must of course
     * have acknowledge of how the data was written out by the save method.
     * You should raise an error in the case that an attempt is being
     * made to load data into an LUT or neural net whose structure does not march
     * the data in the file. (e.g. wrong number of hidden neurons).
     *
     * @param argFileName
     * @throws IOException
     */
    @Override
    public void load(File argFileName) throws IOException {

        BufferedReader read = null;
        try {
            read = new BufferedReader(new FileReader(argFileName));
            for (int i = 0; i < States.numStates; i++)
                for (int j = 0; j < Actions.numActions; j++) {
                    qTable[i][j] = Double.parseDouble(read.readLine());

                }
        } catch (NumberFormatException e) {
            e.printStackTrace();
            initialiseLUT();
        } catch (IOException e) {
            e.printStackTrace();
            System.out.println("IOException trying to open reader: " + e);
            initialiseLUT();
        }catch(NullPointerException e) {
            e.printStackTrace();
            initialiseLUT();
        }

        finally {
            try {
                if (read != null)
                    read.close();
            } catch (IOException e) {
                e.printStackTrace();
                System.out.println("IOException trying to close reader: " + e);
            }catch(NullPointerException e) {
                e.printStackTrace();
            }
        }
    }

    //get the Q_value from Q_table given state and action
    public double getQValue (int state, int action) {
        return qTable[state][action];
    }

    //set the Q_Value for given state and action
    public void setQValue (int state, int action, double qValue) {
        this.qTable[state][action] = qValue;
    }

    //get the max_Q_value for given state
    public double getMaxQValue (int state) {
/*
        double maxQvalue = Double.NEGATIVE_INFINITY;
        for (int actionN = 0; actionN < this.qTable[state].length; actionN++) {
            if(qTable[state][actionN] > maxQvalue) {
                maxQvalue = this.qTable[state][actionN];
            }
        }

        return maxQvalue;
*/

     /*   double maxQValue = 0;
        for (int action = 0; action < this.qTable[state].length; action++)
            maxQValue = Arrays.stream(qTable[action]).max().getAsDouble();

        return  maxQValue;*/

        double max = qTable[state][0];
        for (int i = 0; i < qTable[state].length; i ++) {
            if ( max < qTable[state][i]) {
                max = qTable[state][i];
            }
        }

        return max;

    }

    //get the best_action for given state
    public int getBestAction (int state) {
        //double maxQValue = 0;
        int bestAct = 0;
        double max = qTable[state][0];
        for (int i = 0; i < qTable[state].length; i ++) {
            if ( max < qTable[state][i]) {
                max = qTable[state][i];
                bestAct = i;
            }
        }
        /*for (int action = 0; action < this.qTable[state].length; action++) {
            maxQValue = Arrays.stream(qTable[action]).max().getAsDouble();
            bestAction = action;
        }*/






        return bestAct;


    }


}

