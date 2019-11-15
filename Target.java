package MyRobo;

import java.util.PrimitiveIterator;

public class Target {
    private double targetBearing;
    private double targetBearingRadians;
    private double targetDistance;
    private double targetVelocity;
    private double targetHeadingRadians;
    private double targetEnergy;
    private double targetX;
    private double targetY;

    public Target(double targetBearing, double targetBearingRadians, double targetDistance, double targetVelocity, double targetHeadingRadians, double targetEnergy, double targetX, double targetY) {
        this.targetBearing = targetBearing;
        this.targetBearingRadians = targetBearingRadians;
        this.targetDistance = targetDistance;
        this.targetVelocity = targetVelocity;
        this.targetHeadingRadians = targetHeadingRadians;
        this.targetEnergy = targetEnergy;
        this.targetX = targetX;
        this.targetY = targetY;
    }

    public double getTargetBearing() {
        return targetBearing;
    }

    public void setTargetBearing(double targetBearing) {
        this.targetBearing = targetBearing;
    }

    public double getTargetDistance() {
        return targetDistance;
    }

    public void setTargetDistance(double targetDistance) {
        this.targetDistance = targetDistance;
    }

    public double getTargetVelocity() {
        return targetVelocity;
    }

    public void setTargetVelocity(double targetVelocity) {
        this.targetVelocity = targetVelocity;
    }

    public double getTargetHeadingRadians() {
        return targetHeadingRadians;
    }

    public void setTargetHeadingRadians(double targetHeadingRadians) {
        this.targetHeadingRadians = targetHeadingRadians;
    }

    public double getTargetBearingRadians() {
        return targetBearingRadians;
    }

    public void setTargetBearingRadians(double targetBearingRadians) {
        this.targetBearingRadians = targetBearingRadians;
    }

    public double getTargetEnergy() {
        return targetEnergy;
    }

    public void setTargetEnergy(double targetEnergy) {
        this.targetEnergy = targetEnergy;
    }

    public double getTargetX() {
        return targetX;
    }

    public void setTargetX(double targetX) {
        this.targetX = targetX;
    }

    public double getTargetY() {
        return targetY;
    }

    public void setTargetY(double targetY) {
        this.targetY = targetY;
    }
}
