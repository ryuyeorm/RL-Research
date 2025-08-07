import java.util.Random;
import java.util.Scanner;

public class MultiArmedBanditGame {
     public void run_game(int num_arm, int num_trials) {
        Random random = new Random();
        Scanner scan = new Scanner(System.in);
        System.out.println("Welcome to two armed bandit");

        // variable declaration
        double bound = 100.0;
        double total_reward = 0;
        double[] arm_val = new double[num_arm];
        for(int i = 0; i < num_arm; i++){
            arm_val[i] = random.nextDouble(bound);
            System.out.println(arm_val[i]);
        }


        String input = "";
        while(num_trials > 0){
            System.out.println("Please choose which level to pull(press e to terminate) : ");
            for(int i = 0; i < num_arm; i++){
                System.out.print("| " + i);
            }
            System.out.println("|");

            //take input from the user
            System.out.print("Your input : ");
            input = scan.nextLine().trim().toLowerCase();
            if(input.equals("e")){
                System.out.println("terminated");
                break;
            }

            //parse num arm into Integer
            int input_arm = -1;
            try{
                input_arm = Integer.parseInt(input);

                //input not within range
                if(input_arm > num_arm - 1|| input_arm < 0){
                    throw new NumberFormatException();
                }
            }catch(NumberFormatException e){
                System.out.println("Please insert valid number");
                continue;
            }

            //generate a randomized number based on probability distribution of each lever
            double output_val = random.nextGaussian() * 40 + arm_val[input_arm];
            output_val = Math.max(0, Math.min(100, output_val));
            total_reward += output_val;

            //output, and decrease the remaining trials
            System.out.println("Output value for " + input_arm + ": " + output_val);
            System.out.println("Remaining trials : " + --num_trials+ "\n\n");

        }
        System.out.println("Your total output value is : " + total_reward);
        return;
    }

}
