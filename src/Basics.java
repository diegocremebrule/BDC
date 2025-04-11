import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

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
        return Vectors.dense(1.0,2.0,3.0);
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

    public static void main(String[] args) {
        ArrayTest();
    }

}