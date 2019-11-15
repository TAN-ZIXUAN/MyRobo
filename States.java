package MyRobo;

import javax.swing.plaf.metal.MetalTheme;

public class States {
    public static final int SegDistance2target = 3;    //3 segmentation close:d<=200, medium:200<d<=400, far:d>400
    // private int distanceAfterSeg;

   // public static final int SegEnergy = 4;
   // public static final int SegEnergy_Target = 4;
    public static final int SegGeneralLocation = 3; //0:corner 1:edge 2:centre
    public static final int SegGeneralLocation_Target = 3;

    public static final int SegX = 8;
    public static final int SegY = 6;

    //4 segmentation 0:e<=10, 1:10<e<=25, 2:25<e<=45, 3:e>45
    //but I already consider energy problem in fire so maybe I don't need this case.
    //  private int EnergyAfterSeg;



    //public  static  final  int SegTargetBearing = 4;
   // public static final int SegHitByBullet = 2;

    //public static final int SegTargetBearingRadians = 4;
    public static final int SegAbsBearingRadians = 4; //every 90 degrees

    //public static final int SegHeading = 4;

    public static final int numStates;


    private static final int Mapping[][][][];

    static {
        Mapping = new int[SegDistance2target][SegAbsBearingRadians][SegX][SegY];
        int count = 0;
        for (int a = 0; a < SegDistance2target; a++)
            for (int b = 0; b < SegAbsBearingRadians; b++)
                for (int e = 0; e < SegX; e++)
                    for (int f = 0; f < SegY; f++)
                        Mapping[a][b][e][f]= count++;




        numStates = count;
    }

    public static int distanceAfterSeg(double actualDistance) {
        int d;  //3 segmentation close:d<=200, medium:200<d<=400, far:d>400

        if (actualDistance <= 200)
            d = 0;
        else if (actualDistance <=400)
            d = 1;
        else
            d = 2;

        return d; //return the index for distance
    }

    public static int energyAfterSeg(double actualEnergy) {
        int e;  //4 segmentation 0:e<=10, 1:10<e<=25, 2:25<e<=45, 3:e>45

        if (actualEnergy <= 10)
            e = 0;
        else if (actualEnergy <=25)
            e = 1;
        else if (actualEnergy <= 45)
            e = 2;
        else
            e = 3;

        return e; //return the index for distance
    }

    public static int generalLocationAfterSeg(double x, double y) {
        int index;
        double d1 = distance(x,y,0,0);
        double d2 = distance(x,y,800,0);
        double d3 = distance(x,y,800,600);
        double d4 = distance(x,y,0,600 );
        if(d1<=150||d2<=150||d3<=150||d4<=150) {
            index = 0; //corner
        }
        else if(x<=150||y>=450||y<=150||x>=650) {
            index = 1; // edge
        }
        else
            index = 2;// centre
        return index;

    }

    public static double distance(double x1, double y1, double x2, double y2) {
        double distance;
        double a = x1 - x2;
        double b = y1 - y2;
        a = Math.pow(a,2);
        b = Math.pow(b,2);
        distance = Math.pow((a+b), 0.5);
        return distance;
    }

    public static int absBearingRadiansAfterSeg(double absBearingRadians) {
        int index;
        if(absBearingRadians >0 && absBearingRadians<= 90)
            index = 0;
        else if(absBearingRadians<= 180)
            index = 1;
        else if(absBearingRadians<=270)
            index = 2;
        else
            index = 3;


        return index;
    }

    public static int xAfterSeg(double x) {
        int index;
        if(x<=100)
            index = 0;
        else if (x<=200)
            index = 1;
        else if (x<=300)
            index = 2;
        else if (x<=400)
            index = 3;
        else if (x<=500)
            index = 4;
        else if (x<=600)
            index = 5;
        else if (x<=700)
            index = 6;
        else
            index =7;

        return index;

    }

    public static int yAfterSeg(double y) {
        int index;
        if(y<=100)
            index = 0;
        else if (y<=200)
            index = 1;
        else if (y<=300)
            index = 2;
        else if (y<=400)
            index = 3;
        else if (y<=500)
            index = 4;
        else
            index =5;

        return index;
    }

   /* public static int headingAfterSeg(double heading) {
        double unit = 360 / SegHeading;
        double newHeading = heading + unit / 2;
        if(newHeading > 360.0)
            newHeading -= 360;

        return (int)(newHeading / unit);
    }*/

    public static int getIndexForStates(int distanceIndex,int absBearingRadiansIndex, int xIndex, int yIndex) {
        return Mapping[distanceIndex][absBearingRadiansIndex][xIndex][yIndex];
    }
}
