import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.commons.lang3.tuple.Triple;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class HW1 {

    // A few printing functions that I used along the way
    // To understand what the variables we're working with look like.
    // Feel free to put them into the main method to check anything
    public static void print_data(JavaRDD<String> rdd) {
        rdd.take(10).forEach(System.out::println);
    }

    public static void print_intlist(int[] list) {
        for (int i = 0; i < list.length; i++) System.out.println(list[i]);
    }

    public static void print_listofInteger(List<Integer> list){
        for (int i=0; i<list.size(); i++) System.out.println(list.get(i));
        System.out.println(list.size());
    }
    public static void print_JDDVector(JavaRDD<Vector> rdd) {
        rdd.take(30).forEach(System.out::println);
    }

    public static void print_ArrayofVectors(Vector[] vec) {
        try {
            for (int i = 0; i < 10; i++) System.out.println(vec[i]);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    public static void print_ListofVectors(List<Vector> list) {
        for (Vector point : list) System.out.println(point);
        System.out.println(list.size());
    }

    public static void print_javaRDDdouble(JavaRDD<Double> jrd) {
        System.out.println(jrd.take(10));
    }

    public static void print_double(double dub) {
        System.out.println(dub);
    }

    //Separately processed data in the below method, copied from Nico's
    //It outputs a list of RDD Vectors, [U, A, B]. This makes
    //Using A and B separately easier later on, but need to check
    //with prof if it's okay to change the input slightly from what
    //is described in the assignment, because he says to use U as
    //input for the next functions, and separate into A and B in them.
    public static JavaRDD<Vector>[] CreateDatasets(JavaRDD<String> rdd) {
        JavaRDD<Vector> U = rdd.map(line -> {
            String[] parts = line.split(",");

            double x = Double.parseDouble(parts[0]);
            double y = Double.parseDouble(parts[1]);

            return Vectors.dense(x, y);
        });
        JavaRDD<Vector> A = rdd.filter(line -> line.endsWith("A"))
                .map(line -> {
                    String[] parts = line.split(",");
                    double x = Double.parseDouble(parts[0]);
                    double y = Double.parseDouble(parts[1]);
                    return Vectors.dense(x, y);
                });

        //Group B

        JavaRDD<org.apache.spark.mllib.linalg.Vector> B = rdd.filter(line -> line.endsWith("B"))
                .map(line -> {
                    String[] parts = line.split(",");
                    double x = Double.parseDouble(parts[0]);
                    double y = Double.parseDouble(parts[1]);
                    return Vectors.dense(x, y);
                });

        JavaRDD<Vector>[] datasets = new JavaRDD[3];

        datasets[0] = U;
        datasets[1] = A;
        datasets[2] = B;

        return datasets;

    }

    //Method that trains the Kmeans model and returns the centroids C
    //Need to set a seed for reproducibility
    public static Vector[] Kmeans_C(JavaRDD<Vector> train, int K, int M) {

        JavaRDD<Vector> points = train.map(point -> {
            double x = point.apply(0);
            double y = point.apply(1);
            return Vectors.dense(x,y);
        });

        KMeansModel model = KMeans.train(points.rdd(), K, M);

        //Collect centroids in a list of Vectors

        Vector[] C = new Vector[K];
        for (int i = 0; i < K; i++) C[i] = model.clusterCenters()[i];

        return C;

    }

    public static double MRComputeStandardObjective(JavaRDD<Vector> points, Vector[] C) {
        JavaRDD<Double> dist_sq = points.map(point -> {
            double min = 1000000000;
            for (int i = 0; i < C.length; i++) {
                double dist = Vectors.sqdist(point, C[i]);
                if (dist < min) {
                    min = dist;
                }
            }
            return min;
        });

        double sumsq = dist_sq.reduce((a, b) -> a + b);
        double obj_val = sumsq / points.count();

        return obj_val;

    }

    public static double MRComputeFairObjective(JavaRDD<Vector> A, JavaRDD<Vector> B, Vector[] C) {
        JavaRDD<Double> dist_sqA = A.map(point -> {
            double min = 100000000;
            for (int i = 0; i < C.length; i++) {
                double dist = Vectors.sqdist(point, C[i]);
                if (dist < min) {
                    min = dist;
                }
            }
            return min;
        });

        JavaRDD<Double> dist_sqB = B.map(point -> {
            double min = 100000000;
            for (int i = 0; i < C.length; i++) {
                double dist = Vectors.sqdist(point, C[i]);
                if (dist < min) {
                    min = dist;
                }
            }
            return min;
        });

        double sumA = dist_sqA.reduce((a, b) -> a + b);
        double sumB = dist_sqB.reduce((a, b) -> a + b);

        double objA = sumA / A.count();
        double objB = sumB / B.count();

        double obj_fair_val = Math.max(objA, objB);

        return obj_fair_val;

    }

    //Triple<Integer, Integer, Integer>
    public static void MRPrintStatistics(JavaRDD<Vector> A, JavaRDD<Vector> B, Vector[] C) {

        // List of Vectors containing points in A and B
        List<Vector> A_points = A.collect();
        List<Vector> B_points = B.collect();

        //Debug notes: Variables have expected format down to here.

        //Debug notes: A_points is 782 points.
        //Debug notes: B_points is 230 points.
        // List of Integers, where each entry corresponds to
        // The centroid assigned to the point at that index in A or B
        List<Integer> A_Ci = new ArrayList<>();
        List<Integer> B_Ci = new ArrayList<>();


        for (int i = 0; i < A_points.size(); i++) {
            int My_Ci = 0;
            double min = 100000000;
            for (int j = 0; j < C.length; j++) {
                double dist = Vectors.sqdist(A_points.get(i), C[j]);
                if (dist < min) {
                    min = dist;
                    My_Ci = j;
                }
            }
            A_Ci.add(i, My_Ci);
        }

        for (int i = 0; i < B_points.size(); i++) {
            int My_Ci = 0;
            double min = 100000000;
            for (int j = 0; j < C.length; j++) {
                double dist = Vectors.sqdist(B_points.get(i), C[j]);
                if (dist < min) {
                    min = dist;
                    My_Ci = j;
                }
            }
            B_Ci.add(i, My_Ci);
        }

        //Debug Notes: A_Ci corrected and is a list of integers, of values 0,1,2,3
        //Debug Notes: A_Ci corrected and is correctly size 782
        //Debug Notes: Bug found: Using .set() instead of .add()
        //Debug Notes: Bug found: Looping over C[i] instead of C[j]
        //Debug Notes: B_Ci equally corrected, size 230.
        //Maybe this could be done more efficiently with map()


        int[] A_counts = new int[C.length];
        int[] B_counts = new int[C.length];

        for (int i = 0; i < C.length; i++) {
            A_counts[i] = Collections.frequency(A_Ci, i);
            B_counts[i] = Collections.frequency(B_Ci, i);
        }

        //A_counts correctly a list of integers that sum to 782
        //B_counts correctly " that sum to 230

        List<Triple> Triplets = new ArrayList<>();

        for (int i = 0; i < A_counts.length; i++) {
            Triplets.add(i, Triple.of(i, A_counts[i], B_counts[i]));
        }

        for (int i = 0; i < Triplets.size(); i++) {
            System.out.println(Triplets.get(i));
        }

        //Debug Notes: was using .set() instead of .add()
    }

    // Different Methods for optimisation:

    public static JavaRDD<Vector> CreateDatasets2(JavaRDD<String> rdd){
        JavaRDD<Vector> U = rdd.map(line -> {
           String[] parts = line.split(",");

           double x = Double.parseDouble(parts[0]);
           double y = Double.parseDouble(parts[1]);

            int group;

           if (parts[2].equals("A")){
               group = 0;
           } else {
               group = 1;
           }

           return Vectors.dense(x,y,group);

           });
        return U;
    };

    public static JavaRDD<Vector> CalculateDistances(JavaRDD<Vector> U, Vector[] C){
        JavaRDD<Vector> DCi = U.map(point -> {
           double min = 1000000000;
           int My_C = 0;
           double group = point.apply(2);
           Vector coords = Vectors.dense(point.apply(0), point.apply(1));
           for (int i = 0; i < C.length; i++){
               double dist = Vectors.sqdist(coords, C[i]);
               if (dist < min){
                   min = dist;
                   My_C = i;
               }
           }
           return Vectors.dense(min, My_C, group);
        });
        return DCi;
    }

    public static double MRComputeStandardObjective2(JavaRDD<Vector> DCi){
        JavaRDD<Double> dist_sq = DCi.map(point ->{
            double dist = point.apply(0);
            return dist;
        });

        double sumsq = dist_sq.reduce((a,b) -> a+b);
        double obj_val = sumsq / DCi.count();
        return obj_val;
    }

    public static double MRComputeFairObjective2(JavaRDD<Vector> DCi){
        JavaRDD<Double> A_Sqdist = DCi.filter(line -> line.apply(2) == 0).map(point -> {
            double dist = point.apply(0);
            return dist;
        });

        JavaRDD<Double> B_Sqdist = DCi.filter(line -> line.apply(2)==1).map(point -> {
            double dist = point.apply(0);
            return dist;
        });

        double sumA = A_Sqdist.reduce((a,b) -> a+b);
        double sumB = B_Sqdist.reduce((a,b) -> a+b);

        double objA = sumA / A_Sqdist.count();
        double objB = sumB / B_Sqdist.count();

        double obj_fair_val = Math.max(objA, objB);

        return obj_fair_val;
    }

    //OJO PrintStatistics necesita input K
    public static void MRPrintStatistics2(JavaRDD<Vector> DCi, Vector[] C){
        JavaRDD<Double> A_Ci = DCi.filter(line -> line.apply(2)==0).map(point -> {
            double centroid = point.apply(1);
            return centroid;
        });

        JavaRDD<Double> B_Ci = DCi.filter(line -> line.apply(2)==1).map(point -> {
            double centroid = point.apply(1);
            return centroid;
        });

        int[] NACounts = new int[C.length];
        int[] NBCounts = new int[C.length];

        for (int i = 0; i<C.length; i++){
            Double index = (double) i;
            long Acount = A_Ci.filter(Ci -> (double) Ci == index).count();
            long Bcount = B_Ci.filter(Ci -> (double) Ci == index).count();

            NACounts[i] = (int) Acount;
            NBCounts[i] = (int) Bcount;
        }

        List<Triple> Triplets = new ArrayList<>();

        for (int i = 0; i < C.length; i++){
            Triplets.add(i, Triple.of(i, NACounts[i], NBCounts[i]));
        }

        for (int i = 0; i < Triplets.size(); i++) {
            System.out.println(Triplets.get(i));
        }

    }


    public static void main(String[] args){
        SparkConf conf = new SparkConf()
                .setAppName("KMeansClustering")
                .setMaster("local[*]")
                .set("spark.driver.host", "localhost");

        // Initialize JavaSparkContext
        JavaSparkContext sc = new JavaSparkContext(conf);
        // Read the data into a string RDD
        JavaRDD<String> data = sc.textFile("uber_small.csv");
        //Peep the data
        print_data(data);

        JavaRDD<Vector> U = CreateDatasets2(data);

        Vector[] C = Kmeans_C(U, 4, 20);

        JavaRDD<Vector> DCi = CalculateDistances(U, C);

        double Obj_fun_val = MRComputeStandardObjective2(DCi);
        double Obj_fair_val = MRComputeFairObjective2(DCi);

        print_double(Obj_fun_val);
        print_double(Obj_fair_val);

        MRPrintStatistics2(DCi, C);


        /* Commenting this code out while I test the new methods
        JavaRDD<Vector>[] Input_data = CreateDatasets(data);

        JavaRDD<Vector> U = Input_data[0];
        JavaRDD<Vector> A = Input_data[1];
        JavaRDD<Vector> B = Input_data[2];

        Vector[] C = Kmeans_C(U, 4, 20);

        double Obj_fun_val = MRComputeStandardObjective(U, C);
        double Obj_fair_fun_val = MRComputeFairObjective(A,B,C);

        print_double(MRComputeStandardObjective(U, C));
        print_double(MRComputeFairObjective(A,B,C));

        MRPrintStatistics(A,B,C);

        sc.close();

         */
    }

    /* To check: See that big datasets of similar size to U are
    always stored in RDDs. Check other inefficiencies highlighted
    in the code, as that has to be in the final analysis.
     */
}

