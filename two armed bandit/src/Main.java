import java.util.Random;
import java.util.Scanner;

/**
 * simple program to run trials of multi-armed bandit, with each value of arms randomized, and each value when
 * lever is pulled is sampled from the normal probability distribution with mean of true value of each lever.
 */
public class Main {
    static int num_arm = 3;
    static int num_trials = 1000;
    static MultiArmedBanditGameRL mab = new MultiArmedBanditGameRL(num_arm, num_trials);

    public static void main(String[] args){
        int choice = mab.run_game();
        System.out.println("Best Choice by agent is : " + choice);
        System.out.println("True value for each arm is : ");
        for(int i = 0; i < num_arm; i++){
            System.out.println(mab.arm_val[i]);
        }
    }


}