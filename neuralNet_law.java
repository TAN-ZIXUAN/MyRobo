package MyRobo;


import java.io.*;


public class neuralNet_law {
    double[][] input_vec;                                   //input value vector 1st dimen is how many examples
    double[] inPut;                                         //store one input example

    int input_num;                                          //number of input
    int output_num;                                         //number of output
    double[][] output_vec;                                  //output value vector 1st dimen is how many examples
    double[] outPut;                                        //store one output example
    double[][] neural_node;                                 //store value in the middle neurons
    double[][] delta;                                       //store delta value

    final int depth_num = 3;                                          //how many layers are there
                                                        //this is a NN, with 3 hidden layers,1 input layer, 1 output layer.
    final int[] node_num = {0,14,12,0};                 //regularize how many neurones in each laye

    double[][][] weight;                    //store weight matrix
    double[][][] delta_weight;              //store weight update value
    double rate;                            //learning rate
    double lambda;                           //back step momentum


    File weight_File = new File("/Users/Lawrence Li/robocode/robots/NN_weight.txt"); //where weight stored

    public neuralNet_law(int input_num,int output_num) throws IOException {       //construct func

        this.rate =0.0001;
        this.lambda = 0.9;
        this.input_num = input_num;
        this.output_num = output_num;


        this.neural_node = new double[depth_num+1][];
        this.delta = new double[depth_num][];                                             //initialize delta size

        this.weight = new double[depth_num][][];


        this.neural_node[0] = new double[input_num+1];                          //+1 means this is a bias node

        this.neural_node[0][input_num] = 1;                                     //node_bias = 1


        //set each layer's neuron's length
        for(int i=1;i<depth_num;i++) {                    //layer[0] has been initialized to input[0]

            this.neural_node[i] = new double[node_num[i]+1];
            this.delta[i-1] = new double[neural_node[i].length-1];            //initialize delta size
            this.neural_node[i][this.neural_node[i].length-1] = 1;         //node_bias = 1

        }

        this.neural_node[depth_num] = new double[output_num];  //all neural node initialized
        this.delta[depth_num-1] = new double[output_num];        //initialize delta size

        //set weight size
        for(int i=1;i<depth_num;i++){
            this.weight[i-1] = new double[this.neural_node[i].length-1][this.neural_node[i-1].length];
        }

        this.weight[depth_num-1] = new double[output_num][this.neural_node[depth_num-1].length];
        if(!weight_File.exists()) {
            saveRandomWeight(this.weight, -1, 1);
        }//give this net a initial random weight
        else load(weight_File);
        this.delta_weight = equals_size(this.weight);        //initial delta

    }       //构造函数，构造一个神经网络。深度、每层神经数在此设定。

    public static double getRandomDouble(double min, double max){
        return ((double) (Math.random()*(max - min))) + min;
    }      //取得一个随机数

    public static void getRandomWeight(double[][][] wei,double min, double max){

        for(int i = 0;i<wei.length;i++){
            for(int j = 0;j<wei[i].length;j++){
                for(int k=0;k<wei[i][j].length;k++){
                    wei[i][j][k] = getRandomDouble(min,max);
                }
            }
        }
    }       //初始化 weight 矩阵
    public static void saveRandomWeight(double[][][] wei,double min, double max){

        PrintStream fout =null;
        FileOutputStream pout =null;
        try {
            pout = new FileOutputStream("NN_weight.txt",true);
            fout = new PrintStream(pout,true);
            for (int i = 0; i < wei.length; i++) {
                for (int j = 0; j < wei[i].length; j++) {
                    for (int k = 0; k < wei[i][j].length; k++) {
                        wei[i][j][k] = getRandomDouble(min, max);
                        fout.println(wei[i][j][k]);
                    }
                }
            }
        }
        catch(IOException e){e.printStackTrace();}
        finally {
            fout.close();
        }

    }




    public static void getZeroWeight(double[][][] wei){

        for(int i = 0;i<wei.length;i++){
            for(int j = 0;j<wei[i].length;j++){
                for(int k=0;k<wei[i][j].length;k++){
                    wei[i][j][k] = 0;
                }
            }
        }

    }   //初始化 一个 全为0的 矩阵



    public double nero_sum(double[] input,double[] w){
        int len_input = input.length;
        int len_w = w.length;
        double nero_value = 0;
        if (len_input != len_w){
            System.out.println("length no match!");
        }
        else{
            for(int i = 0;i<len_input;i++){
                nero_value += input[i]*w[i];
            }
        }
        return nero_value;
    }           //向量内积

    //Three activate functions
    public double Sigmoid(double x){ return (1 / (1 + Math.exp(-x))); }
    public double tanh(double x){ return (Math.exp(2*x)-1)/(Math.exp(2*x)+1); }
    public double tanh_deri(double x){return 1-x*x;}
    public double ReLu(double x){ return Math.max(0.0f,x);}
    public double Leaky_ReLu(double x){
        return Math.max(0.01*x,x);
    }
    public double ReLu_deri(double x){
        if(x > 0){return 1;}
        else {return 0;}
    }

    public static double Leaky_ReLu_deri(double x){
        if(x>0){return 1;}
        else {return 0.01;}
    }
    //Sigmoid for vector
    public double[] Sigmoid(double[] x){
        double[] sx =new double[x.length];
        for(int i=0;i<x.length;i++){
            sx[i] = Sigmoid(x[i]);
        }
        return sx;
    }

    public double[][][] equals_size(double[][][] v){
        double[][][] w = new double[v.length][][];

        for(int i=0;i<v.length;i++){
            w[i] = new double[v[i].length][];
            for(int j = 0;j<v[i].length;j++){
                w[i][j] = new double[v[i][j].length];
            }
        }
        return w;
    }

    public double[][][] equals_value(double[][][] v){
        double[][][] w = new double[v.length][][];

        for(int i=0;i<v.length;i++){
            w[i] = new double[v[i].length][];
            for(int j = 0;j<v[i].length;j++){
                w[i][j] = new double[v[i][j].length];
            }
        }
        for(int i = 0;i<v.length;i++){
            for(int j = 0;j<v[i].length;j++){
                for(int k=0;k<v[i][j].length;k++){
                    w[i][j][k] = v[i][j][k];
                }
            }
        }

        return w;
    }

    public double[][] equals_value(double[][] v){
        double[][] w = new double[v.length][];
        for(int i=0;i<v.length;i++){
            w[i] = new double[v[i].length];
        }
        for(int i = 0;i<v.length;i++){
            for(int j=0;j<v[i].length;j++){
                w[i][j] = v[i][j];
            }
        }
        return w;
    }




    public void updateWeight(int i){     //update i th layer's weights with momentum
       double[][] ow = equals_value(weight[i]);
       for (int j = 0; j < ow.length; j++) {
           for (int k = 0; k <ow[j].length ; k++) {
               double dw = lambda* delta_weight[i][j][k];
               weight[i][j][k] = ow[j][k] + dw + rate * delta[i][j] * neural_node[i][k];
               //delta_weight[i][j][k] = rate * delta[i][j] * neural_node[i][k];
               delta_weight[i][j][k] = weight[i][j][k]-ow[j][k];
           }
       }
   }

    public void updateDelta(int i) {                        //update the i th layer delta
        double wei_sum = 0;
        double yj = 0;
        double d = 0;

        if(i == depth_num-1){                               //if it is the top layer
            for(int j=0;j<delta[i].length;j++){
                delta[i][j] = Leaky_ReLu_deri(neural_node[depth_num][j])
                        * (outPut[j]-neural_node[depth_num][j]);
            }
        }
        if(i < depth_num-1 && i>0){
            for(int j=0;j<delta[i].length;j++) {
                wei_sum = 0;
                for (int m = 0; m < weight[i + 1].length; m++) {
                    wei_sum += weight[i + 1][m][j] * delta[i+1][m];
                }
                delta[i][j] = wei_sum * Leaky_ReLu_deri(neural_node[i+1][j]);
            }
        }
        if(i == 0){
            for(int j=0;j<delta[i].length;j++) {
                wei_sum = 0;
                for (int m = 0; m < weight[i + 1].length; m++) {
                    wei_sum += weight[i + 1][m][j] * delta[i+1][m];
                }
                delta[i][j] = wei_sum * tanh_deri(neural_node[i+1][j]);
                //delta[i][j] = wei_sum * ReLu_deri(neural_node[i+1][j]);
            }
        }

    }

    public void update(){
        for (int i = depth_num-1; i >= 0 ; i--) {
            updateDelta(i);
            updateWeight(i);
        }
    }

    public double[] outPut1(double[] inPut){     //compute output from input
        double[]outPut = new double[this.output_num];
        int l = inPut.length;
        int ld = this.depth_num-1;
        double whatever = 0;
        for(int i = 0;i<l;i++){
            this.neural_node[0][i] = inPut[i];
        }
        for (int i = 0; i < weight[0].length; i++) {
            this.neural_node[1][i] = tanh(nero_sum(this.neural_node[0],weight[0][i]));
            //this.neural_node[1][i] = ReLu(nero_sum(this.neural_node[0],weight[0][i]));
        }//from input to first hidden layer, use sigmoid func. Try to quantize the state;
        for(int d=2;d<this.depth_num;d++){
            for( int j=0;j<this.weight[d-1].length;j++) {
                this.neural_node[d][j] = Leaky_ReLu(nero_sum(this.neural_node[d-1],weight[d-1][j]));
            }
        }
        for( int j=0;j<this.neural_node[this.depth_num].length;j++) {
            this.neural_node[this.depth_num][j] = Leaky_ReLu(nero_sum(this.neural_node[ld],this.weight[ld][j]));
        }

        for (int k = 0; k < this.output_num; k++) {
            whatever = this.neural_node[this.depth_num][k];
            outPut[k] = whatever;
        }

        return outPut;
    }

    public double[][] outPut2(double[][] inPut){        //compute output vector from input vector
        int l_in = inPut.length;
        int l_ou = this.outPut.length;
        double[][] outp = new double[l_in][l_ou];
        double[] whatever = new double[l_ou];
        for(int i = 0;i<l_in;i++){
            whatever = outPut1(inPut[i]);
            outp[i] = whatever;
        }
        return outp;
    }



    public double lossFunc(double[][] y,double[][] yhat){
        double loss=0;
        for(int i=0;i<y.length;i++) {
            for(int j=0;j<y[i].length;j++)
            loss += 0.5*Math.pow((y[i][j]-yhat[i][j]),2.0);
        }
        return loss;
    }

    public static void print_wei(double[][][] wei){
        for(int i=0;i<wei.length;i++){
            for(int j=0;j<wei[i].length;j++){
                for(int k=0;k<wei[i][j].length;k++){
                    System.out.println("weight ("+i+","+j+","+k+") is: "+wei[i][j][k]);
                }
            }
        }
    }

    public void trainNet(double[][] X,double[][] C,int iterations){


        for (int j = 0; j < iterations; j++) {
            for (int k = X.length - 1; k >= 0; k--) {
                this.inPut = X[k];
                this.outPut = C[k];
                outPut1(inPut);
                update();
            }//fork
        }//forj
        save(weight_File);
    }

    public void load(File file) throws IOException {
        BufferedReader reader = new BufferedReader(new
                FileReader(file));
        String line = "";
        try {
                for (int i = 0; i < weight.length; i++) {
                    for (int j = 0; j < weight[i].length; j++) {
                        for (int k = 0; k < weight[i][j].length; k++) {
                            line = reader.readLine();
                            if(line != null) {
                                weight[i][j][k] = Double.parseDouble(line);
                            }
                        }
                    }
                }

        } catch (IOException e) {
            e.printStackTrace();
        }finally {
            reader.close();
        }
    }//load

    public void save(File file) {

        PrintStream w = null;

        try {
            w = new PrintStream(new FileOutputStream(file));
            for (int i = 0; i < weight.length; i++) {
                for (int j = 0; j < weight[i].length; j++) {
                    for (int k = 0; k < weight[i][j].length; k++) {
                        w.println(weight[i][j][k]);
                    }
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }finally {
            w.flush();
            w.close();
        }

    }//save

    //given input, return the index of the output which has the maximum value
    public int max_output(double[] v){
        double[] op = outPut1(v);
        int max_index = 0;
        double tmp = op[max_index];
        for (int j = 0; j < op.length; j++) {
            if(op[j]>tmp){max_index = j;}
        }

        return max_index+1;
    }

    public static double[] double_combine(double[] a,double[] b){
        int la = a.length;
        int lb = b.length;

        double[] r = new double[la+lb+1];
        System.arraycopy(a,0,r,0,la);
        System.arraycopy(b,0,r,la,lb);

        return r;
    }





    public static void main(String[] args) throws IOException {
        double[][] in = {{-1,-1},{-1,1},{1,-1},{1,1}};
        double[][] in2 = {{2,1},{10,5},{5,7},{99,45}};
        double[][] in1 = {{0,0},{0,1},{1,0},{1,1}};
        double[][] in_test = {{4,10,9,4,0.05},{10,20,6,0.8,10},{23,5,5,7,0.12},{32,2,3,9,70}};
        double[][] ou = {{-1},{1},{1},{-1}};
        double[][] ou2 = {{3},{2},{1},{4}};
        double[][] ou_test = {{0.2,0.8,0.4,0.3},{1.6,0.4,1,0.1},{0.22,0.95,2,0.5},{0.5,0.2,0.6,1.4}};
        double[][] ou_test2 = {{0.2},{1.6},{2},{5}};

        double[] loss= new double[1000];
        neuralNet_law nn = new neuralNet_law(10,1);

        //nn.rate = 0.2;
        //getRandomWeight(nn.weight,-0.5,0.5);
        //nn.delta_weight = equals_size(nn.weight);


        double[] ttt = nn.outPut1(new double[]{-1,-1});
        double[] ttt1 = nn.outPut1(new double[]{1,-1});
        ttt = nn.outPut1(new double[]{4,10,9,4,0.05});
        ttt1 = nn.outPut1(new double[]{32,2,3,9,70});

        System.out.println(ttt[0]+","+ttt1[3]);
        nn.trainNet(in_test,ou_test,9990000);
        ttt = nn.outPut1(new double[]{4,10,9,4,0.05});
        ttt1 = nn.outPut1(new double[]{32,2,3,9,70});
         //ttt = nn.outPut1(new double[]{1,1});
         //ttt1 = nn.outPut1(new double[]{-1,1});
        System.out.println(ttt[2]+","+ttt1[2]);
        //System.out.println(nn.max_output(new double[]{32,2,3,9,70}));




    }
}
