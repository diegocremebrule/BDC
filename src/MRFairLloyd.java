import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;

public static Vector[] MRFairLloyd(JavaRDD<Tuple2<Vector, String>> U, int K, int M) {
        JavaRDD<Vector> vector = U.map(p -> p._1); //get a vector of our points, taking out the 1st tuple.
        KMeansModel model = KMeans.train(vector.rdd(), K, iterations=0); // here we do step1: set {c1,c2,…,ck} of k centroids
        Vector[] centroids = model.clusterCenters(); //get the coordinates of our centroids

        for (int i = 0; i < M; i++) { // we start for loop until M times
            JavaRDD<Tuple2<Integer, Tuple2<Vector, String>>> idx_tuple = U.map(coord_label -> {
                Vector coords = coord_label._1;  // get the coordinates of the points
                String label = coord_label._2;  // get the label (A/B)

                // get the index of the centroid which is closest to coordinates using function idx_close created
                int idx = idx_close(coords, centroids);

                // the map should return the following tuple: (idx, (coordinates, label))
                return new Tuple2<>(idx, new Tuple2<>(coords,label));
            });
            //add new lists of A and B and inside of them create K lists that we will add the respective points
            //we return a map: centroid index -> (Vector, Label)
            Map<Integer, Iterable<Tuple2<Vector, String>>> clustered = idx_tuple.groupBy(Tuple2::_1).collectAsMap();

            //lists A and B for each cluster (k clusters total)
            List<List<Vector>> A = new ArrayList<>();
            List<List<Vector>> B = new ArrayList<>();
            for (int i = 0; i < K; i++) {
                A.add(new ArrayList<>()); //empty lists so clutser goes in position i of A and B respectively
                B.add(new ArrayList<>());

                 //important!! this gets the list of all data points assigned to cluster i 
                Iterable<Tuple2<Vector, String>> items = clustered.getOrDefault(i, new ArrayList<>());

                //now we start to iterate over the points in cluster i and then assign the point
                //to the corresponding cluster list for A and B
                for (Tuple2<Vector, String> tuple : items) {
                    if (tuple._2.equals("A")) {
                        A.get(i).add(tuple._1);
                    } else {
                        B.get(i).add(tuple._1);
                    }
                }
            }
            // apply function of centroids in which we give A and B vectors listed with the points for each cluster
            //0 to k. So A=[cluster1:[(2,3),(4,5)], cluster2:....]
            //this function is below and chooses the right centroids according to our problem needs.
            centroids = CentroidSelection(A, B);
        }

        return centroids;
    }

//this function gets the index of the centroid which is closest to coordinates, calculating the sqdist
public static int idx_close(Vector coordinates, Vector[] centroids) {
    double min = Double.MAX_VALUE;
    int idx = -1;

    for (int i = 0; i < centroids.length; i++) {
        double dist = Vectors.sqdist(coordinates, centroids[i]);

        if (dist < min) {
            min = dist;
            idx = i;
        }
    }

    return idx;
}

// Function to compute alpha, beta, MA, MB, and L given groups A, B and number of clusters k for CentroidSelection
//we use result that helps us bring multiple results of the same function.
public static Result computeVectors(List<List<Vector>> A, List<List<Vector>> B, int k) {
        // Calculate total number of points in group A and group B
    
        int total_A = 0;
        for (List<Vector> clustered_A : A) { // for loop cluster's points in A
            // Add the number of points in this cluster to the total
            total_A = total_A + clustered_A.size();
        }

        int total_B = 0;

       
        for (List<Vector> clustered_B : B) { //for loop cluster's points in A
            // Add the number of points in this cluster to the total
            total_B = total_B + clustered_B.size();
        }

        double[] alfa = new double[k]; //initialize alfa
        double[] beta = new double[k]; //initialize beta
        Vector[] MA = new Vector[k]; //initialize MA
        Vector[] MB = new Vector[k]; //initialize MB
        double[] l = new double[k]; //initialize l

        // loop within the clusters
        for (int i = 0; i < k; i++) {
            List<Vector> clusterA = A.get(i);
            List<Vector> clusterB = B.get(i);

            int sizeA = clusterA.size();
            int sizeB = clusterB.size();

            alfa[i] = (double) sizeA / total_A; //compute alfa
            beta[i] = (double) sizeB / total_B; //compute beta

            //Compute vector averages for A using function below that calculates these averages per cluster
            MA[i] = meanVector(clusterA);

            //Compute vector averages for B using function below that calculates these averages per cluster
            MB[i] = meanVector(clusterB);

            // If MA[i] is null (no points in A for cluster i) -> MA[i] = MB[i]
            if (MA[i] == null && MB[i] != null) {
                MA[i] = MB[i];
            }

            // If MB[i] is null (no points in B for cluster i) -> MB[i] = MA[i]
            if (MB[i] == null && MA[i] != null) {
                MB[i] = MA[i];
            }

            // Euclidean distance to complete L
            if (MA[i] == null || MB[i] == null) {
                L[i] = 0; // No distance if centroids undefined
            } else {
                L[i] = Math.sqrt(Vectors.sqdist(MA[i], MB[i]));
            }
        }

        // Return all vectors need for function Centroid Selection
        return new Result(alpha, beta, MA, MB, L);
    }

    //function to compute mean vector of a list of vectors, calculates MA and MB
    private static Vector meanVector(List<Vector> points) {
        if (points.isEmpty()) return null;

        int dim = points.get(0).size();
        double[] sum = new double[dim];

        for (Vector v : points) {
            double[] arr = v.toArray();
            for (int j = 0; j < dim; j++) {
                sum[j] += arr[j];
            }
        }

        for (int j = 0; j < dim; j++) {
            sum[j] /= points.size();
        }

        return Vectors.dense(sum);
    }

//we use result that helps us bring multiple results of the same function. In this case we bring the results
//of function computeVectors, so we can implement them in CentroidSelection
public static class Result {public double[] alfa;public double[] beta;public Vector[] MA;public Vector[] MB;public double[] L;

    public Result(double[] alfa, double[] beta, Vector[] MA, Vector[] MB, double[] L) {
        this.alfa = alfa;
        this.beta = beta;
        this.MA = MA;
        this.MB = MB;
        this.L = L;
    }
}

//Centroid Selection that receives vector A,B, k clusters and moreover receives the computed vectors plus
//the function that computes Vector x.
public static Vector[] CentroidSelection(List<List<Vector>> A, List<List<Vector>> B, int k) {
    //Step 1 compute vectors
    Result r = computeVectors(A, B, k);
    double[] alpha = r.alfa;
    double[] beta = r.beta;
    Vector[] MA = r.MA;
    Vector[] MB = r.MB;
    double[] L = r.L;

    //step 2 calculate first denominator in fixedA and fixedB
    int total_A = A.stream().mapToInt(List::size).sum();
    int total_B = B.stream().mapToInt(List::size).sum();

    //Step 2 calculate first numerator delta(A, MA) and delta(B, MB)
    double deltaA = 0;
    for (int i = 0; i < k; i++) {
        for (Vector a : A.get(i)) {
            deltaA += Vectors.sqdist(a, MA[i]);
        }
    }

    double deltaB = 0;
    for (int i = 0; i < k; i++) {
        for (Vector b : B.get(i)) {
            deltaB += Vectors.sqdist(b, MB[i]);
        }
    }

    //Step 2 calculate fixedA and fixedB
    double fixedA = deltaA / total_A;
    double fixedB = deltaB / total_B;

    //Step 3 obtain vector x from function
    double[] x = computeVectorX(fixedA, fixedB, alpha, beta, L, k);

    //Step 4 obtain centroids 
    Vector[] centroids = new Vector[k];
    for (int i = 0; i < k; i++) {
        if (L[i] == 0) {
            centroids[i] = MA[i];
        } else {
            double xi = x[i];
            double li = L[i];
            double coefA = (li - xi) / li;
            double coefB = xi / li;

            double[] MAarr = MA[i].toArray();
            double[] MBarr = MB[i].toArray();
            double[] cArr = new double[MA[i].size()];

            for (int d = 0; d < MA[i].size(); d++) {
                cArr[d] = coefA * MAarr[d] + coefB * MBarr[d];
            }
            centroids[i] = Vectors.dense(cArr);
        }
    }

    return centroids;
}

