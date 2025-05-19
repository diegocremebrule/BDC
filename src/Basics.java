import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import java.io.File;
import java.io.IOException;
import java.sql.SQLInvalidAuthorizationSpecException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import scala.Tuple2;

public class Basics {
    public static void initial() {
        int number = 10;
        double pi = 3.14;
        String greet = "Hello world!";

        System.out.println(number);
        System.out.println(pi);
        System.out.println(greet);
    }

    public static void saludo() {
        System.out.println("Wena po chala zico silla de playa lagartija vegana zapato de colegio languetazo completo de Talca");
    }

    public static void InputPractice() throws IOException{
        File mytxt = new File("Test txt.txt");
        Scanner reader = new Scanner(mytxt);
        while(reader.hasNextLine()){
            String data = reader.nextLine();
            System.out.println(data);
        }

    }

    public static void forloop(){
        for (int i=0; i<5; i++){
            System.out.println(i);
        }
    }

    public static void whileloop(){
        int j=0;
        while(j<10){
            ++j;
            System.out.println(j);
        }
    }

    public static Vector CreateVector(){
        JavaRDD<Vector> A = java.util.Vector.dense(1,2,3);
    }

    public static void ListLength(){
        int[] test = new int[4];
        test[0] = 10;
        test[1] = 20;
        test[2] = 15;
        test[3] = 40;
        System.out.println(test);
    }

    public static void ArrayTest(){
        List<Integer> test1 = new ArrayList<>();
        test1.add(10);
        test1.add(20);
        test1.add(30);
        test1.add(40);

        System.out.println(test1.get(0));
        System.out.println(test1.get(1));
        System.out.println(test1.get(2));

        test1.set(0,30);
        test1.set(1,40);
        test1.set(2,50);

        System.out.println(test1.get(0));
        System.out.println(test1.get(1));
        System.out.println(test1.get(2));
    }

    public static Vector ComputeVectorX(double fixedA, double fixedB, Vector alpha, Vector beta, Vector dist, int k, double T ){
        double gamma = 0.5;

        double[] ArrayX = new double[k];
        
        for (int t=0; t<T; t++){
            double fAx = fixedA;
            double fBx = fixedB;

            for (int i=0; i<k; k++){
              double alphai = alpha.apply(i);
              double betai = beta.apply(i);
              double disti = dist.apply(i);

              double xi = ((1-gamma)*betai*disti)/(gamma*alphai+(1-gamma)*betai);

              ArrayX[i] = xi;

              fAx = fAx + alphai*xi*xi;
              fBx = fBx + betai*(disti-xi)*(disti-xi);

              if (fAx == fBx){
                break;
              } else{
                if (fAx > fBx){
                    gamma = gamma + Math.pow(0.5, t+1);
                } else {
                    gamma = gamma - Math.pow(0.5, t+1);
                }
              }

            }

            }

        Vector X = Vectors.dense(ArrayX);

        return X;

        }


    public static void main(String[] args) {
        ArrayTest();
        saludo();
    }

}