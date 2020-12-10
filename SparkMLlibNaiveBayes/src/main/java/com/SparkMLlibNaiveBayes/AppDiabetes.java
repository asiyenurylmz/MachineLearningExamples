package com.SparkMLlibNaiveBayes;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.classification.NaiveBayesModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class AppDiabetes {
	public static void main(String[] args) {
		SparkSession sparkSession = SparkSession.builder().master("local").appName("diabetes-mllib").getOrCreate();
		Dataset<Row> rawData = sparkSession.read().format("csv").option("header", true).option("inferschema", true)
				.load("C:\\Users\\asiye\\OneDrive\\Masaüstü\\datasets\\diabetes.csv");

		String[] headerList = { "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
				"DiabetesPedigreeFunction", "Age", "Outcome" };

		List<String> headers = Arrays.asList(headerList);
		List<String> headersResult = new ArrayList<String>();
		for (String h : headers) {
			if (h.equals("Outcome")) {
				StringIndexer indexTmp = new StringIndexer().setInputCol(h).setOutputCol("label");
				rawData = indexTmp.fit(rawData).transform(rawData);
				headersResult.add("label");
			} else {
				StringIndexer indexTmp = new StringIndexer().setInputCol(h).setOutputCol(h.toLowerCase() + "_cat");
				rawData = indexTmp.fit(rawData).transform(rawData);
				headersResult.add(h.toLowerCase() + "_cat");
			}
		}

		String[] colList = headersResult.toArray(new String[headersResult.size()]);

		VectorAssembler vectorAssembler = new VectorAssembler().setInputCols(colList).setOutputCol("features");

		Dataset<Row> transformData = vectorAssembler.transform(rawData);

		Dataset<Row> finalData = transformData.select("label", "features");

		Dataset<Row>[] dataSets = finalData.randomSplit(new double[] { 0.9, 0.1 });

		Dataset<Row> trainData = dataSets[0];
		Dataset<Row> testData = dataSets[1];

		NaiveBayes nb = new NaiveBayes();
		nb.setSmoothing(1);
		NaiveBayesModel model = nb.fit(trainData);

		Dataset<Row> predictions = model.transform(testData);

		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator().setLabelCol("label")
				.setMetricName("accuracy");

		double evaluate = evaluator.evaluate(predictions);
		
		predictions.show();
		System.out.println("Accuracy : " + evaluate);

	}
}