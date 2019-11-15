package MyRobo;

public class Actions {
    //public enum actions {circle, retreat, advance, fire};
    /*
    circle: circle around enemy
    retreat: step back
    advance: step ahead
    fire: fire at the enemy
     */

    //not working well. robot couldn't learn with complicate actions.
    /*public static final int robotSpiral = 5;
    public static final int robotTrack = 6;
    //public static final int robotTrack1 =7;
    //public static final int robotRetreat =8;*/

    //public static final int rotate1 = 5;

    public static final int robotAhead = 0;
    public static final int robotBack = 1;
    public static final int robotTurnLeft = 2;
    public static final int robotTurnRight = 3;
    public static final int robotSpin = 4;
    public static final int robotFire = 5; //fire with max bullet power:3

    /*public static final int robotAhead_L = 5;
    public static final int robotBack_L = 6;
    public static final int robotTurnLeft_L = 7;
    public static final int robotTurnRight_L = 8;*/


    public static final int numActions =6;

    public static final double RobotMoveDistance = 100.0;
    public static final double RobotTurnDegree =  90.0;

    public static final double RobotMoveDistance_L = 5000;
    public static final double RobotTurnDegree_L =  180.0;





}
