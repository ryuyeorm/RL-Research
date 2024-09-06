import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;


class NeuralNetwork{
    int hidden_num = 4;
    int input_num = 2;
    double[][] w1 = new double[2][hidden_num];
    double[] w2 = new double[hidden_num];
    double[] b1 = new double[hidden_num];
    double b2;

    static Random rand = new Random();
    //initial
    public NeuralNetwork(){
        for(int j = 0; j < hidden_num; j++){
            for(int l = 0; l < input_num; l++){
                w1[l][j] = 2 * rand.nextDouble() - 1;
            }
            w2[j] = 2 * rand.nextDouble() - 1;
            b1[j] = 2 * rand.nextDouble() - 1;
        }
        b2  = 2 * rand.nextDouble() - 1;
    }
    //offspring
    public NeuralNetwork(NeuralNetwork p1, NeuralNetwork p2){
        //randomly select weight and biases
        for(int j = 0; j < hidden_num; j++){
            for(int l = 0; l < input_num; l++){
                if(rand.nextBoolean()){
                    w1[l][j] = p1.w1[l][j];
                }else{
                    w1[l][j] = p2.w1[l][j];
                }
            }
            if(rand.nextBoolean()){
                w2[j] = p1.w2[j];
            }else{
                w2[j] = p2.w2[j];
            }

            if(rand.nextBoolean()){
                b1[j] = p1.b1[j];
            }else{
                b1[j] = p2.b1[j];
            }
        }
        if(rand.nextBoolean()){
            b2 = p1.b2;
        }else{
            b2 = p2.b2;
        }
    }

    public int compute(int x1, int x2){
        double[] x = new double[]{x1, x2};
        double[] h = new double[hidden_num];
        double o = 0.0;

        //forward pass
        for(int j = 0; j < 3; j++) {
            for (int l = 0; l < 2; l++) {
                h[j] += x[l] * w1[l][j];
            }
            h[j] = sigmoid(h[j] + b1[j]);
        }
        for(int j = 0; j < 3; j++)
            o += h[j] * w2[j];
        o = sigmoid(o + b2);

        return binary(o);
    }

    public void mutate(){
        double m_rate = 0.1;
            for(int j = 0; j < hidden_num; j++){
                for(int l = 0; l < input_num; l++){
                    if(rand.nextDouble() < m_rate){
                        if(rand.nextBoolean()){
                            w1[l][j] *= rand.nextDouble()/2.0;
                        }else{
                            w1[l][j] /= rand.nextDouble()/2.0;
                        }
                    }
                }
                if(rand.nextDouble() < m_rate){
                    if(rand.nextBoolean()){
                        w2[j] *= rand.nextDouble()/2.0;
                    }else{
                        w2[j] /= rand.nextDouble()/2.0;
                    }
                }

                if(rand.nextDouble() < m_rate) {
                    if(rand.nextBoolean()){
                        b1[j] *= rand.nextDouble()/2.0;
                    }else{
                        b1[j] /= rand.nextDouble()/2.0;
                    }
                }
            }
            if(rand.nextDouble() < m_rate){
                if(rand.nextBoolean()){
                    b2 *= rand.nextDouble()/2.0;
                }else{
                    b2 /= rand.nextDouble()/2.0;
                }
            }
        }

    private double sigmoid(double z){
        return 1 / (1 + Math.exp(-z));
    }
    private int binary(double o){
        return (int)Math.round((o * 2) / 2);
    }

}


public class Main {
    final static int NUM_NN = 80;

    static NeuralNetwork[] NN = new NeuralNetwork[NUM_NN];
    static Random rand = new Random();
    static int[] fitness = new int[NUM_NN];

    public static void main(String[] args) throws IOException {

        for(int i = 0; i < NUM_NN; i++){
            NN[i] = new NeuralNetwork();
        }
        setObs();
        for(int i = 0; i < 40; i++){
            System.out.println("ITERATION : " + i);
            simulateGame();
            update();
        }

    }

    private static void update(){
        NeuralNetwork[] new_NN = new NeuralNetwork[NUM_NN];
        for(int i = 0 ; i < NUM_NN - 1; i++){
            int parent1 = random_selection();
            int parent2 = random_selection();
            //create offspring
            new_NN[i] = new NeuralNetwork(NN[parent1], NN[parent2]);
            System.out.println(parent1 + " : " + parent2);
        }
        int max_idx = -1;
        int max = Integer.MIN_VALUE;
        for(int i = 0; i < fitness.length; i++){
            if(fitness[i] > max){
                max = fitness[i];
                max_idx = i;
            }
        }
        int elite = max_idx;
        new_NN[NUM_NN - 1] = NN[elite];
        System.arraycopy(new_NN, 0, NN, 0, new_NN.length);
        for(int i = 0; i < NUM_NN; i++){
            NN[i].mutate();
        }
    }

    private static int random_selection(){
        //select random parent with score as weight
        int min = Arrays.stream(fitness).min().getAsInt();
        if(min < 0){
            int total = Arrays.stream(fitness).sum() - min * fitness.length;
            double randTotal = rand.nextDouble() * total;
            for(int i = 0; i < NUM_NN; i++){
                randTotal -= fitness[i];
                randTotal -= -min;
                if(randTotal <= 0){
                    return i;
                }
            }
        }else{
            int total = Arrays.stream(fitness).sum();
            double randTotal = rand.nextDouble() * total;
            for(int i = 0; i < NUM_NN; i++){
                randTotal -= fitness[i];
                if(randTotal <= 0){
                    return i;
                }
            }
        }
        int total = Arrays.stream(fitness).sum();
        double randTotal = rand.nextDouble() * total;
        for(int i = 0; i < NUM_NN; i++){
            randTotal -= fitness[i];
            if(randTotal <= 0){
                return i;
            }
        }
        System.out.println("Rand total ?? : " + randTotal + " : " + Arrays.toString(fitness));
        return -1;
    }

    static ArrayList<Integer> obs = new ArrayList<>();
    private static void setObs(){       //random opening of the obstacle.
        for(int i = 0 ; i < 1000; i++){
            obs.add((int) (17 + Math.round(rand.nextDouble() * 60)));
        }
    }

    private static void simulateGame() throws IOException {
        int pos_x = 0;
        int[] pos_y = new int[NUM_NN];
        Arrays.fill(pos_y, 50);
        boolean[] jump = new boolean[NUM_NN];
        boolean[] alive = new boolean[NUM_NN];
        Arrays.fill(alive, true);
        boolean gameOver = false;

        //game start
        while(!gameOver){
            pos_x++;
            for(int agent = 0; agent < NUM_NN; agent++){
                if(pos_x > 20000){
                    System.out.println("agent " + agent + " passed : " + pos_x + "!!!!!!!!!!!!!!!!");
                    File file = new File("weight2.txt");
                    FileWriter writer = new FileWriter(file);
                    writer.write(Arrays.deepToString(NN[agent].w1) + "\n");
                    writer.write(Arrays.toString(NN[agent].w2) + "\n");
                    writer.write(Arrays.toString(NN[agent].b1) + "\n");
                    writer.write(Double.toString(NN[agent].b2));
                    writer.close();
                    return;
                }
                //if agent is not alive, skip
                if(!alive[agent]){continue;}
                //if jump pressed last frame, then update the y position accordingly
                if(jump[agent]){
                    pos_y[agent] = (pos_y[agent] + 10)%100;
                }else{
                    pos_y[agent]--;
                }
                if(pos_y[agent] <= 0){
                    pos_y[agent] = 0;
                }


                //check collision, and save score if collided
                if(pos_x % 17 == 0 && (pos_y[agent] <= obs.getFirst() || pos_y[agent] >= (obs.getFirst() + 15))){
                    alive[agent] = false;
                    fitness[agent] = pos_x - Math.abs(obs.getFirst() + 8 - pos_y[agent]);
                }else if(pos_x % 17 == 0){
                }
                //compute if need to jump using NN
                if(alive[agent]){
                    jump[agent] = NN[agent].compute(17 - (pos_x % 17), pos_y[agent] - obs.getFirst() - 8) == 1;
                }
            }
            //check if at least one agent is alive
            gameOver = true;
            for(int i = 0; i < NUM_NN; i++){
                if (alive[i]) {
                    gameOver = false;
                    break;
                }
            }
            //remove obstacle that agent has passed, and add another obstacle
            if(pos_x % 17 == 0){
                obs.removeFirst();
                obs.add((int) (17+ Math.round(rand.nextDouble() * 40)));
            }
        }

    }

}