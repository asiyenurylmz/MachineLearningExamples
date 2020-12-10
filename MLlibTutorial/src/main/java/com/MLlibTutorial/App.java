package com.MLlibTutorial;

import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.LinearRegressionTrainingSummary;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

/**
 * Hello world!
 *
 */
public class App {
	public static void main(String[] args) {
		System.setProperty("hadoop.home.dir", "C:\\Users\\asiye\\OneDrive\\Masaüstü\\hadoop-common-2.2.0-bin-master");
		SparkSession sparkSession = SparkSession.builder().appName("spark-mllib").master("local").getOrCreate();

		Dataset<Row> rawData = sparkSession.read().format("csv").option("header", "true").option("inferSchema", "true")
				.load("C:\\Users\\asiye\\OneDrive\\Masaüstü\\datasets\\satis.csv");

		Dataset<Row> newData = sparkSession.read().format("csv").option("header", "true").option("inferSchema", "true")
				.load("C:\\Users\\asiye\\OneDrive\\Masaüstü\\datasets\\test.csv");

		VectorAssembler featuresVector = new VectorAssembler().setInputCols(new String[] { "Ay" }) // Girdi verileri
																									// string dizisi
																									// olarak burada
																									// verilir--Bağımsız
																									// değişkenlerin
																									// dizisi
				.setOutputCol("features"); // Giriş verileri dizisinin vector e dönüşmüş çıktısı

		Dataset<Row> transform = featuresVector.transform(rawData);

		Dataset<Row> transformNewData = featuresVector.transform(newData);

		Dataset<Row> finalData = transform.select("features", "Satis");

		Dataset<Row>[] datasets = finalData.randomSplit(new double[] { 0.7, 0.3 }); // 0.7 Train Data 0.3 Test Data

		Dataset<Row> trainData = datasets[0];
		Dataset<Row> testData = datasets[1];

		LinearRegression lr = new LinearRegression();
		lr.setLabelCol("Satis"); // Tahmin etmesi istenenilen kolon

		LinearRegressionModel model = lr.fit(trainData); // Train Data 'dan linear regression modeli oluşturuyoruz

		LinearRegressionTrainingSummary summary = model.summary();	//Bu sınıf modelin başarı oranını ölçebileceğimiz methodlara sahip
		
		System.out.println(summary.r2());

		Dataset<Row> transformTest = model.transform(testData); // testData verilerinin model e göre tahmini yapılıyor

		// Dataset<Row> transformTest = model.transform(transformNewData);
		// transformTest.show(30);

	}
}
