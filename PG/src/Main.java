import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

class NeuralNetwork {
    double lr = 0.001;
    double prev_loss = 0;
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
        for (int k = 0; k < output_num; k++) {
            for (int j = 0; j < hidden_num; j++)
                w2[j][k] = 2 * rand.nextDouble() - 1;
            b2[k] = 2 * rand.nextDouble() - 1;
        }
    }

    public double[] compute(int x1, int x2) {
        x[0] = x1;
        x[1] = x2;
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

        o = softmax(o);

        return o;
    }
    public void copy(NeuralNetwork NN){

    }

    public double[] fd_calculate(int state_x, int state_y, int id, int[] idx, double q, int eps_length){
        //finite difference
        double epsilon = 0.00001;
        NeuralNetwork[] eps_nn = new NeuralNetwork[2];
        for(int i = 0; i < 2; i++){
            System.arraycopy(eps_nn[i].w1, 0, w1, 0, w1.length);
            System.arraycopy(eps_nn[i].w2, 0, w2, 0, w1.length);
            System.arraycopy(eps_nn[i].b1, 0, b1, 0, b2.length);
            System.arraycopy(eps_nn[i].b2, 0, b2, 0, b2.length);
        }
        //[0, w1][1,w2][2,w3][3,w4]
        switch (id){
            case 0:
                eps_nn[0].w1[idx[0]][idx[1]] += epsilon;
                eps_nn[1].w1[idx[0]][idx[1]] -= epsilon;
                break;
            case 1:
                eps_nn[0].w2[idx[0]][idx[1]] += epsilon;
                eps_nn[1].w2[idx[0]][idx[1]] -= epsilon;
                break;
            case 2:
                eps_nn[0].b1[idx[0]] += epsilon;
                eps_nn[1].b1[idx[0]] -= epsilon;
                break;
            case 3:
                eps_nn[0].b2[idx[0]] += epsilon;
                eps_nn[1].b2[idx[0]] -= epsilon;
                break;
        }

        //calculate finite difference
        double[] val_1 = eps_nn[0].compute(state_x, state_y);
        double[] val_2 = eps_nn[1].compute(state_x, state_y);

        double loss_1 = -(1.0/eps_length) * q  * Math.log(pr_a)



        double[] out = new double[2];
        double[] loss_fd = new double[2];
        for(int i = 0; i < 2; i++){
            out[i] = (val_1[i] - val_2[i])/(epsilon * 2.0);
        }

        return out;

        /*
        //compute gradient respect to w1[0][0]
        double[] pr_a = compute(state_x.get(time), state_y.get(time));
        double[] val_grad_1 = new double[2];
        for(int k = 0; k < output_num; k++) {
            val_grad_1[k] +=  (pr_a[k] - ((a.get(time) == 1)?1:0)) * w2[y_idx][a.get(time)] * h[y_idx] * (1 - h[y_idx]) * x[x_idx];
        }
        System.out.println(Arrays.toString(out));
        System.out.println(Arrays.toString(val_grad_1));
        */
    }

    public void train(ArrayList<Integer> state_x, ArrayList<Integer> state_y, ArrayList<Integer> a, ArrayList<Integer> r) {
        System.out.println("_______________________next eps+_____________________-");
        double discount = 0.95;
        int eps_length = state_x.size();

        double[] Q = new double[eps_length];
        double[] loss = new double[eps_length];

        //calculating Q value
        for (int t = eps_length - 1; t >= 0; t--) {
            Q[t] = r.get(t) + (t < eps_length - 1 ? discount * Q[t + 1] : 0);

        }
        //calculating Loss
        for (int t = 0; t < eps_length; t++) {
            double pr_a = compute(state_x.get(t), state_y.get(t))[a.get(t)];
            loss[t] = -(1.0/eps_length) * Q[t] * Math.log(pr_a); // loss = - (1/t) * Qt * log(pr_a)
        }

        double loss_batch = Arrays.stream(loss).sum();
        System.out.println("previous loss : " + prev_loss + " current loss : " + loss_batch + " : episode length " + eps_length);
        prev_loss = loss_batch;

        for(int t = 0; t < eps_length; t++){
            for(int i = 0; i < 4; i++){
                for(int l = 0; l < input_num; l++){
                    for(int j = 0; j < hidden_num; j++){
                        int[] idx = {l, j};
                        fd_calculate(state_x.get(t), state_y.get(t), 0, idx, Q[t], eps_length);
                    }

                }
            }
        }
        double [] grad = fd_calculate(state_x, state_y, a, r);

        /*


        //calculate gradient for second weight & biases
        double[][] grad_w2 = new double[hidden_num][output_num];
        double[] grad_b2 = new double[output_num];
        double[] y_i = new double[output_num]; // set y_i[k]

        //for each episode
        for (int t = 0; t < eps_length; t++) {
            double[] pr_a = compute(state_x.get(t), state_y.get(t));
            int action = a.get(t);
            y_i[action] = 1;
            //for each j, k add up the gradient of each second layer weight
            for (int k = 0; k < output_num; k++) {
                for (int j = 0; j < hidden_num; j++) {
                    grad_w2[j][k] += Q[t] * (pr_a[k] - y_i[k]) * h[j]; // gradient of loss should be - (1/t) Q(t) * ( delta and should be subtracted
                }
                grad_b2[k] += Q[t] * (pr_a[k] - y_i[k]);
            }
            grad_b2[action] += Q[t] * (pr_a[action] - y_i[action]);
        }
        // for each second layer weight and biases subtract the gradient divided by the -episode length and learning rate of 0.01

        for (int k = 0; k < output_num; k++) {
            for (int j = 0; j < hidden_num; j++) {
                w2[j][k] += lr * ((double) 1 / eps_length) * grad_w2[j][k];
            }
            b2[k] += lr * ((double) 1 / eps_length) * grad_b2[k];
        }

        y_i = new double[output_num];

        double[] grad_b1 = new double[hidden_num];
        double[][] grad_w1 = new double[input_num][hidden_num];


        for (int t = 0; t < eps_length; t++) {
            int action = a.get(t);
            y_i[action] = 1;
            double[] pr_a = compute(state_x.get(t), state_y.get(t));
            for(int k = 0; k < output_num; k++){
                for (int j = 0; j < hidden_num; j++) {
                    for (int l = 0; l < input_num; l++) {
                        grad_w1[l][j] += Q[t] * (pr_a[k] - y_i[k]) * w2[j][a.get(t)] * h[j] * (1 - h[j]) * x[l];
                    }
                    grad_b1[j] += Q[t] * (pr_a[k] - y_i[k]) * w2[j][a.get(t)] * h[j] * (1 - h[j]);
                }
            }

        }

        for (int j = 0; j < hidden_num; j++) {
            for (int l = 0; l < input_num; l++) {
                w1[l][j] += ((double) 1 /eps_length) *  lr * grad_w1[l][j] ;
            }
            b1[j] += ((double) 1 /eps_length) * lr * grad_b1[j];
        }

         */
    }

    private double sigmoid(double z) {
        return 1 / (1 + Math.exp(-z));
    }

    private double[] softmax(double[] o) {
        double[] output = new double[o.length];
        double sum = 0;
        for (double v : o) sum += Math.exp(v);
        for (int i = 0; i < o.length; i++) {
            output[i] = Math.exp(o[i]) / sum;
        }
        return output;
    }
}

public class Main {
    static Random rand = new Random();
    static int MAX_EPISODES = 2;

    //replay buffer

    static ArrayList<Integer> obs = new ArrayList<>();

    public static void main(String[] args) throws IOException {
        //create a neural network
        NeuralNetwork NN = new NeuralNetwork();

        //create initial dataset
        setObs();

        //fill experience replay buffer
        for (int i = 0; i < MAX_EPISODES; i++) {
            simulateGame(NN);
        }

        //output the neural network
        FileWriter writer = new FileWriter("weight.txt");
        writer.write(Arrays.deepToString(NN.w1) + "\n");
        writer.write(Arrays.deepToString(NN.w2) + "\n");
        writer.write(Arrays.toString(NN.b1) + "\n");
        writer.write(Arrays.toString(NN.b2) + "\n");
        writer.close();
    }

    private static void setObs() {       //random opening of the obstacle.
        for (int i = 0; i < 1000; i++) {
            if (!obs.isEmpty()) {
                obs.clear();
            }
            obs.add((int) (20 + Math.round(rand.nextDouble() * 60)));
        }
    }

    private static void simulateGame(NeuralNetwork NN) {
        ArrayList<Integer> stateX = new ArrayList<>();
        ArrayList<Integer> stateY = new ArrayList<>();
        ArrayList<Integer> action = new ArrayList<>();
        ArrayList<Integer> reward = new ArrayList<>();

        int pos_x = 0;
        int pos_y = 50;
        int jump = 0;

        //game start
        while (true) {
            pos_x++;
            //if jump pressed last frame, then update the y position accordingly
            if (jump == 1) {
                pos_y = (pos_y + 10) % 100;
            } else {
                pos_y--;
            }
            if (pos_y <= 0) {
                pos_y = 0;
            }

            stateX.add(20 - (pos_x % 20));
            stateY.add(pos_y - obs.get(0) - 10);
            action.add(jump);

            //check collision, and save score if collided
            if (pos_x % 20 == 0 && (pos_y <= obs.get(0) || pos_y >= (obs.get(0) + 20))) {
                reward.add(0);
                NN.train(stateX, stateY, action, reward);
                return;
            } else if(pos_x % 20 == 0){
                reward.add(1000);
            }else{
                reward.add(1);
            }


            double[] o = NN.compute(20 - (pos_x % 20), pos_y - obs.get(0) - 10);

            //random jump
            if (rand.nextDouble(1) < o[0]) {
                jump = 0;
            } else {
                jump = 1;
            }

            //update the obstacle
            if (pos_x % 20 == 0) {
                obs.remove(0);
                obs.add((int) (20 + Math.round(rand.nextDouble() * 40)));
            }


        }
    }

}