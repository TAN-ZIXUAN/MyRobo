package MyRobo.Interface;

import java.io.File;
import java.io.IOException;

public interface CommonInterface {

    /**
     * @param x the input vector. An array of doubles.
     * @return The value returned by the LUT or NN for this input vector
     */

    public double outputFor(double[] x);

    /**
     * This method will tell the NN or the LUT the output
     * value that should be mapped to the given input vector. I.e.
     * the desired correct output value for an input.
     * @param x The input vector
     * @param argValue The new Value to learn
     * @return The error in the output for that input vector
     */

    public double train(double[] x, double argValue);

    /**
     * x:input vector
     * argvalue:desired output
     * A method to write either a LUT or weights of an neural net to a file.
     * @param argFile of type File.
     */

    public void save(File argFile);

    /**
     * Loads the LUT or neural net weights from file. The load must of course
     * have acknowledge of how the data was written out by the save method.
     * You should raise an error in the case that an attempt is being
     * made to load data into an LUT or neural net whose structure does not march
     * the data in the file. (e.g. wrong number of hidden neurons).
     * @throws IOException
     */

    public void load(File argFileName) throws IOException;
}
