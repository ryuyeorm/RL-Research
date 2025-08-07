import javax.swing.plaf.multi.MultiLabelUI;
import java.util.Random;
import java.util.Scanner;

public class MultiArmedBanditGameRL {
    int num_arm;
    int num_trials;
    double bound;
    double total_reward;
    double[] arm_val;
    Random random;

    public MultiArmedBanditGameRL(int num_arm, int num_trials){
        bound = 100.0;
        total_reward = 0;
        this.num_arm = num_arm;
        this.num_trials = num_trials;
        arm_val = new double[num_arm];
        random = new Random();
        for(int i = 0; i < num_arm; i++){
            arm_val[i] = random.nextDouble(bound);
            System.out.println(arm_val[i]);
        }
    }
     public int run_game() {
        Random random = new Random();
        Model model = new Model(num_arm, num_trials);
        System.out.println("Welcome to two armed bandit");

        int input_arm;
        while(num_trials > 0){
            //take input from the user
            input_arm = model.nextChoice();
            System.out.print("Agent's input : ");
            System.out.println(input_arm);


            //generate a randomized number based on probability distribution of each lever
            double output_val = random.nextGaussian() * 30 + arm_val[input_arm];
            output_val = Math.max(0, Math.min(100, output_val));
            total_reward += output_val;

            model.update(input_arm, output_val);

            //output, and decrease the remaining trials
            System.out.println("Output value for " + input_arm + ": " + output_val);
            num_trials--;

        }
        System.out.println("Your total output value is : " + total_reward);

        return model.bestChoice();
    }

}
