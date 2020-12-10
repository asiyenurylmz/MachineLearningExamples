package com.SparkMLlibNaiveBayes;

import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.classification.NaiveBayesModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class App {

	public static void main(String[] args) {

		SparkSession sparkSession = SparkSession.builder().master("local").appName("spark-mllib-naive-bayes")
				.getOrCreate();

		Dataset<Row> loadData = sparkSession.read().format("csv").option("header", true).option("inferschema", true)
				.load("C:\\Users\\asiye\\OneDrive\\Masaüstü\\datasets\\basketbol.csv");

		StringIndexer indexHava = new StringIndexer().setInputCol("hava").setOutputCol("hava_cat"); // String verileri
																									// matematiksel
																									// olarak ifade
																									// edebilmek için
																									// indexliyoruz
		StringIndexer indexSicaklik = new StringIndexer().setInputCol("sicaklik").setOutputCol("sicaklik_cat");
		StringIndexer indexNem = new StringIndexer().setInputCol("nem").setOutputCol("nem_cat");
		StringIndexer indexRuzgar = new StringIndexer().setInputCol("ruzgar").setOutputCol("ruzgar_cat");
		StringIndexer indexLabel = new StringIndexer().setInputCol("basketbol").setOutputCol("label"); // tahmin
																										// edilecek
																										// kolon->label

		Dataset<Row> transformHava = indexHava.fit(loadData).transform(loadData); // loadData verilerini 'hava' için
																					// indexliyoruz
		Dataset<Row> transformSicaklik = indexSicaklik.fit(transformHava).transform(transformHava);
		Dataset<Row> transformNem = indexNem.fit(transformSicaklik).transform(transformSicaklik);
		Dataset<Row> transformRuzgar = indexRuzgar.fit(transformNem).transform(transformNem);
		Dataset<Row> transformResult = indexLabel.fit(transformRuzgar).transform(transformRuzgar);

		VectorAssembler vectorAssembler = new VectorAssembler()
				.setInputCols(new String[] { "hava_cat", "sicaklik_cat", "nem_cat", "ruzgar_cat", "label" })
				.setOutputCol("features"); // Giriş
											// kolonlarını
											// birleştirerek
											// vector
											// haline
											// getiriyoruz

		Dataset<Row> transform = vectorAssembler.transform(transformResult);

		Dataset<Row> finalData = transform.select("label", "features"); // Yalnızca matematiksel ifade ettiğimiz
																		// kolonları alıyoruz

		Dataset<Row>[] dataSets = finalData.randomSplit(new double[] { 0.7, 0.3 });
		Dataset<Row> trainData = dataSets[0];
		Dataset<Row> testData = dataSets[1];

		NaiveBayes nb = new NaiveBayes();
		nb.setSmoothing(1);
		NaiveBayesModel model = nb.fit(trainData); // Eğitiyoruz
		Dataset<Row> predictions = model.transform(testData);

		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator().setLabelCol("label")
				.setPredictionCol("prediction").setMetricName("accuracy"); // Naive Bayes doğruluğunun değerlendirmesini yapıyoruz

		double evaluate = evaluator.evaluate(predictions);

		predictions.show();
		System.out.println("Accuracy = " + evaluate);

	}
}
