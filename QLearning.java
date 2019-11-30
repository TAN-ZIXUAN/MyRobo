package MyRobo;

import java.util.Random;

public class QLearning {
    public static double alpha = 0.2; //learning rate: alpha
    public static double gamma = 0.8; //discount rate: gamma fire has delay rewards
    public static double epsilon = 0.2; //epsilon-greedy: epsilon for exploration, 1-epsilon for exploitation
/*    private int curtState;
    private int curtAction;*/


    private LUT lut;

    //Constructor

    public QLearning(LUT lut) {
        this.lut = lut;
    }

    //Q-learning Algorithm
    public void Q_Learning(int prevState, int prevAction, int crtState, double reward) {
        double QValue;
        double newQValue;

        QValue = lut.getQValue(prevState, prevAction);
        newQValue = QValue + alpha * (reward + gamma * lut.getMaxQValue(crtState) - QValue);
        lut.setQValue(prevState, prevAction, newQValue); //update LUT Q value

    }

    //on-policy learning
    public void Sarsa( int prevState, int prevAction,int crtState, int crtAction, double reward) {
        double Q1;
        double Q2;

            Q1 = lut.getQValue(prevState, prevAction);
            Q2 = lut.getQValue(crtState,crtAction);

            double newQValue = Q1 + alpha*(reward + gamma *Q2 - Q1);
            lut.setQValue(prevState, prevAction, newQValue);

    }

    //policy: epsilon-greedy
    public int policySelectAction(int state) {
        double e = Math.random();
        int action = 0;
        if (e < epsilon) {// exploration
            Random random = new Random();
            action = random.nextInt(Actions.numActions);
        }

        else {
            //exploitation
            action = lut.getBestAction(state);
        }
        return action;
    }

    public static double getAlpha() {
        return alpha;
    }

    public static void setAlpha(double alpha) {
        QLearning.alpha = alpha;
    }

    public static double getGamma() {
        return gamma;
    }

    public static void setGamma(double gamma) {
        QLearning.gamma = gamma;
    }

    public static double getEpsilon() {
        return epsilon;
    }

    public static void setEpsilon(double epsilon) {
        QLearning.epsilon = epsilon;
    }
}
