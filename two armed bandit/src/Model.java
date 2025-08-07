import java.util.Random;

public class Model {
    //simple epsilon greedy algorithm to find optimal choice for agent
    int num_arm;
    int num_trials;
    double decay_rate;
    double eps_g;
    double[] history;
    int[] each_trial;
    /**
     *  constructor for Model Class
     * @param num_arm
     * @param num_trials
     */
    public Model(int num_arm, int num_trials){
        this.num_arm = num_arm;
        this.num_trials = num_trials;
        decay_rate = 1.0/num_trials;
        eps_g = 1.0;
        each_trial = new int[num_arm];
        history = new double[num_arm];
    }

    Random random = new Random();


    /**
     * return next choice of lever to pull based on epsilon greedy value
     * @return integer value of next chosen lever
     */
    public int nextChoice(){
        boolean is_explore = random.nextDouble() < eps_g;
        eps_g -= decay_rate;
        if(is_explore){
            System.out.println("explore !");
            return random.nextInt(num_arm);
        }else{
            System.out.println("Exploit!");
            return bestChoice();
        }
    }

    /**
     * Finds the optimal choice of lever by calculating the average of each arm from exerience
     * @return the current optimal choice lever number
     */
    public int bestChoice(){
        int max_idx = -1;
        double max_val = -1;
        for(int i = 0; i < num_arm; i++){
            if(each_trial[i] > 0){
                double avg_val = history[i]/each_trial[i];
                if(avg_val > max_val){
                    max_val = avg_val;
                    max_idx = i;
                }
            }
        }
        if(max_idx == -1){
            return random.nextInt(num_arm);
        }
        return max_idx;
    }

    public void update(int input_arm, double output_val){
        each_trial[input_arm]++;
        history[input_arm] += output_val;
    }


}
