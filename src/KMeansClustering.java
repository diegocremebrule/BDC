import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;

public class KMeansClustering {
    public static void main(String[] args) {
        SparkConf conf = new SparkConf()
                .setAppName("KMeansClustering")
                .setMaster("local[*]")
                .set("spark.driver.host", "localhost");

        // Initialize JavaSparkContext
        JavaSparkContext sc = new JavaSparkContext(conf);

        JavaRDD<String> data = sc.textFile("uber_small.csv");

        //Change to Vector, we use the map function that helps us transform each element p,g in an RDD
        //we change a String as what we have as input and change it into a Vector of Spark type
        //Data is an RDD that has each row as a String, which means each row of the file can be processed in parallel
        // within different machines.
        //The map function is applying a function that splits our points from the demographic group and returns a vector
        //Getting an RDD with vectors, each points is saved as a vector with .dense
        JavaRDD<Vector> points = data.map(line -> {
            String[] parts = line.split(",");

            double x = Double.parseDouble(parts[0]);
            double y = Double.parseDouble(parts[1]);

            // Convert to Spark Vector
            return Vectors.dense(x, y);
        });

        //Now getting RDD with Vectors for each of the demographic groups

        // Group A

        JavaRDD<Vector> A = data.filter(line -> line.endsWith("A"))
                .map(line -> {
                    String[] parts = line.split(",");
                    double x = Double.parseDouble(parts[0]);
                    double y = Double.parseDouble(parts[1]);
                    return Vectors.dense(x, y);
                });

        //Group B

        JavaRDD<Vector> B = data.filter(line -> line.endsWith("B"))
                .map(line -> {
                    String[] parts = line.split(",");
                    double x = Double.parseDouble(parts[0]);
                    double y = Double.parseDouble(parts[1]);
                    return Vectors.dense(x, y);
                });

        //Kmeans

        int K = 4;  // clusters
        int M = 50;  // iterations

        // KMeans, .rdd() necessary conversion to use the function to train our KMeans.
        KMeansModel model = KMeans.train(points.rdd(), K, M);

        // Retrieve CENTROIDS in Vector type
        Vector[] C = model.clusterCenters();

        // Return obj function, see function below
        double finalValue = ObjectiveFun(points, C);

        //Return obj fair function
        double finalValueFair = FairFunction(A,B,C);

        // Print results
        System.out.println("Final Objective Function Value: " + finalValue);
        System.out.println("Final Fair Objective Function Value: " + finalValueFair);
        System.out.println("K:" + K + " M: " + M + " C: " + C);
        // Stop Spark
        sc.stop();
    }
    // NOW THE OBJECTIVE FUNCTION
    public static double ObjectiveFun(JavaRDD<Vector> points, Vector[] C){
        //First we would calculate the squared distances with the map function, we assign a really large number,
        // compare it and update it when we find a smaller distance. So, we return for the map function the minimum
        // squared distance for each point
        //Basically applying a map function to each point. This function is the calculation of the minimum distance
        //between centroids and the point. For every point in "points" in this case, return the min dist.
        //remember that points now is an RDD of Vectors, each point a Vector
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

        // Then by using the reduce function, we take pair by pair adding them. So if our sq distances are (8,2,3,4)
        // then it does 8+2=10+3=13+4=17
        double total = dist_sq.reduce((a, b) -> a + b);
        // just the average
        double avg = total/ points.count();

        return avg;
    }

    //NOW THE FAIR FUNCTION
    public static double FairFunction(JavaRDD<Vector> A, JavaRDD<Vector> B, Vector[] C){

        // Fair function just evaluating A and B separately with our standard objective function and then
        // choose the max value
        double resultadoA = ObjectiveFun(A,C);
        double resultadoB = ObjectiveFun(B,C);

        double resfinal = Math.max(resultadoA,resultadoB);

        return resfinal;
    }
}
