import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

class NeuralNetwork {
    double discount = 0.01;
    int hidden_num = 10;
    int input_num = 2;
    int output_num = 2;
    double[][] w1 = new double[input_num][hidden_num];
    double[][] w2 = new double[hidden_num][output_num];
    double[] b1 = new double[hidden_num];
    double[] b2 = new double[output_num];
    double[] x = new double[output_num];
    double[] h = new double[hidden_num];
    double[] o = new double[output_num];
    static Random rand = new Random();

    //initial
    public NeuralNetwork() {
        for (int j = 0; j < hidden_num; j++) {
            for (int l = 0; l < input_num; l++) {
                w1[l][j] = 2 * rand.nextDouble() - 1;
            }
            b1[j] = 2 * rand.nextDouble() - 1;
        }
        for(int k = 0; k < output_num; k++){
            for(int j =0; j < hidden_num; j++)
                w2[j][k] = 2 * rand.nextDouble() - 1;
            b2[k] = 2 * rand.nextDouble() - 1;
        }
    }

    public void copy(NeuralNetwork NN){
        System.arraycopy(NN.w1, 0, w1, 0, w1.length);
        System.arraycopy(NN.w2, 0, w2, 0, w1.length);
        System.arraycopy(NN.b1, 0, b1, 0, b2.length);
        System.arraycopy(NN.b2, 0, b2, 0, b2.length);
    }

    public double[] compute(int x1, int x2) {
        x[0] = x1;
        x[1]= x2;
        Arrays.fill(h, 0);
        Arrays.fill(o, 0);
        //forward pass
        for (int j = 0; j < hidden_num; j++) {
            for (int l = 0; l < input_num; l++) {
                h[j] += x[l] * w1[l][j];
            }

            h[j] = sigmoid(h[j] + b1[j]);
        }
        for (int j = 0; j < hidden_num; j++)
            for (int k = 0; k < output_num; k++) {
                o[k] += h[j] * w2[j][k];
            }
        //return index of the greater q value
        return o;
    }

    public void train(int[][] s, int[] a, int[] r, NeuralNetwork target){
        double prev_loss = 0;
        for(int i = 0; i < s.length - 1; i++){
            //estimated value of q by online network
            double q_estimate =  compute(s[i][0], s[i][1])[a[i]];

            //actual reward at state s + discounted estimated value of q by target network at s+1.
            double q_target = r[i] + discount * target.compute(s[i][0], s[i][1])[a[i]];
            //calculate loss
            double loss = Math.pow(q_estimate - q_target, 2);
            double loss_grad = 2 * (q_estimate - q_target);

            System.out.println("prev_loss : " + prev_loss + ", new_loss : " + loss);
            prev_loss = loss;

            //no gradient for absolute value of zero
            if(loss != 0){
                //gradient descent
                for (int j = 0; j < hidden_num; j++) {
                    for (int l = 0; l < input_num; l++) {

                        w1[l][j] -= discount * loss_grad * w2[j][a[i]]* h[j]* (1 - h[j]) * x[l];
                        w2[j][l] -= discount * loss_grad * h[j];
                    }
                    b1[j] -= discount * loss_grad * h[j] * (1 - h[j]);
                }
                for(int l = 0; l < output_num; l++){
                    b2[l] -= discount * loss_grad;
                }
            }

        }

    }

    private double cross_entropy(double[] o, double[] v){
        double sum = 0;
        for(int i = 0; i < o.length; i++){
            sum += -v[i] * Math.log(o[i]);
        }
        return sum;
    }


    private double sigmoid(double z) {
        return 1 / (1 + Math.exp(-z));
    }

    private int binary(double o) {
        return (int) Math.round((o * 2) / 2);
    }

}

public class Main {
    static Random rand = new Random();
    static int buffer_size = 10000;
    static int sample_size = 32;
    //replay buffer
    static int[] reward = new int[buffer_size];
    static int[][] state = new int[buffer_size][2];
    static int[] action = new int[buffer_size];

    static int[] sample_r = new int[sample_size];
    static int[][] sample_s = new int[sample_size][2];
    static int[] sample_a = new int[sample_size];

    static ArrayList<Integer> obs = new ArrayList<>();

    public static void main(String[] args) throws IOException {
        //create a neural network
        NeuralNetwork NN = new NeuralNetwork();

        //create a copy of the neural network (target network)
        NeuralNetwork target_NN = new NeuralNetwork();
        target_NN.copy(NN);

        //create initial dataset
        setObs();

        //fill experience replay buffer
        simulateGame(NN, target_NN);

        //output the neural network

        FileWriter writer = new FileWriter("weight.txt");
        writer.write(Arrays.deepToString(NN.w1) + "\n");
        writer.write(Arrays.deepToString(NN.w2) + "\n");
        writer.write(Arrays.toString(NN.b1) + "\n");
        writer.write(Arrays.toString(NN.b2) + "\n");
        writer.close();

    }

    private static void sample(int timestep){
        for(int i = 0; i < sample_size; i++){
            int idx = rand.nextInt(timestep);
            sample_r[i] = reward[idx];
            sample_s[i] = state[idx];
            sample_a[i] = action[idx];
        }
    }

    private static void setObs() {       //random opening of the obstacle.
        for (int i = 0; i < 1000; i++) {
            if(!obs.isEmpty()){
                obs.clear();
            }
            obs.add((int) (20 + Math.round(rand.nextDouble() * 60)));
        }
    }

    private static void simulateGame(NeuralNetwork NN, NeuralNetwork target_NN){
        int pos_x = 0;
        int pos_y = 50;
        int jump = 0;
        int time = 0;
        //game start
        while (true) {
            time++;
            pos_x++;
            if(time >= buffer_size){
                return;
            }
            //if jump pressed last frame, then update the y position accordingly
            if (jump == 1) {
                pos_y = (pos_y + 10) % 100;
            } else {
                pos_y--;
            }
            if (pos_y <= 0) {
                pos_y = 0;
            }
            state[time][0] = 20 - (pos_x % 20);
            state[time][1] = pos_y - obs.get(0) - 10;
            action[time] = jump;

            //check collision, and save score if collided
            if (pos_x % 20 == 0 && (pos_y <= obs.get(0) || pos_y >= (obs.get(0) + 20))) {
                reward[time] = 0;
                System.out.println(pos_x / 20);
                if(time < buffer_size){
                    pos_x = 0;
                    pos_y = 50;
                    jump = 0;
                    setObs();
                    continue;
                }
            }else{
                reward[time] = 1;
            }

            if(time > 256){
                sample(time);
                NN.train(sample_s, sample_a, sample_r, target_NN);
            }

            if(time % 1000 == 0){
                NN.copy(target_NN);
            }

            //compute if need to jump using NN
            double[] o = NN.compute(20 - (pos_x % 20), pos_y - obs.get(0) - 10);

            //eps-greedy to choose action between jump and not jump
            double eps = 0.1;
            if(rand.nextDouble() < eps){
                if(rand.nextBoolean())
                    jump = 1;
                else
                    jump = 0;
            }else
                jump = (o[1] > o[0]) ? 1 : 0;

            //update the obstacle
            if (pos_x % 20 == 0) {
                obs.remove(0);
                obs.add((int) (20 + Math.round(rand.nextDouble() * 40)));
            }


        }
    }

}