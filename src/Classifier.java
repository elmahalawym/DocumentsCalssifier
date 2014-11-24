import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.net.MalformedURLException;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;

import jnisvmlight.LabeledFeatureVector;
import jnisvmlight.SVMLightModel;

import java.sql.*;


public class Classifier {
	
	private static int POLITICS_LABEL = 4;
	private static int SPORTS_LABEL = 2;
	private static int TECH_LABEL = 3;
	private static int GENERAL_LABEL = 1;
	
	private static String SPORTS_MODEL_FILENAME = "Models//sportsModel";
	private static String POLITICS_MODEL_FILENAME = "Models//politicsModel";
	private static String TECH_MODEL_FILENAME = "Models//techModel";
	static String FEATURES_FILENAME = "\featureToIndex.txt";
	
	public final static String ARABIC_CHARACHTERS = "اأإآبتثجحخدذرزسشصضطظعغفقكلمنهويىئءؤة";
	public final static String NON_ARABIC_CHARACHTER = "[^" + ARABIC_CHARACHTERS + "]";
	
	public static double THRESHOLD = 0;
	
	public static void main(String[] args) throws IOException, ParseException {					

		// read featureToIndex from file.
		System.out.println("reading featureToIndex...");
		HashMap<String, Integer> featureToIndex = readFeatureToIndex();
		
		// reading model files.
		System.out.println("reading model files...");
		SVMLightModel sportsModel = readModel(SPORTS_MODEL_FILENAME);
		SVMLightModel politicsModel = readModel(POLITICS_MODEL_FILENAME);
		SVMLightModel techModel = readModel(TECH_MODEL_FILENAME);

		// example: classifying an article in test.txt
		// reading article
		BufferedReader br = new BufferedReader(new FileReader("test.txt"));
		String file = readDocument("test.txt");
		br.close();
		int result = Classify(file, politicsModel, sportsModel, techModel, featureToIndex);
		
		System.out.println("test document is classified as " + result);

	}
	
	
	/*
	 * Classifies an article 
	 */
	public static int Classify(String article, SVMLightModel politicsModel, 
			SVMLightModel sportsModel, SVMLightModel techModel,
			HashMap<String, Integer> featureToIndex) {
		
		double politicsScore = evaluateModel(article, politicsModel, featureToIndex);
		double sportsScore = evaluateModel(article, sportsModel, featureToIndex);
		double technologyScore = evaluateModel(article, techModel, featureToIndex);
		return detectClass(politicsScore, sportsScore, technologyScore);
	}
	
	
	
	///////// PRIVATE METHODS ////////////////////
	
	
	@SuppressWarnings("deprecation")
	private static SVMLightModel readModel(String modelfileName) throws MalformedURLException, ParseException {
	
		File file = new File(modelfileName);
		
		SVMLightModel model = SVMLightModel.readSVMLightModelFromURL(file.toURL());
		
		return model;
		
	}
	
	
	// reads featureToIndex
	private static HashMap<String, Integer> readFeatureToIndex() throws IOException {
		
		HashMap<String, Integer> featureToIndex = new HashMap<>();
		String fileName = "featureToIndex.txt";
		BufferedReader br = new BufferedReader(new FileReader(fileName));
		String curLine;
		
		while((curLine = br.readLine()) != null) {
			String[] line = curLine.split(" ");
			featureToIndex.put(line[0], Integer.parseInt(line[1]));
		}
		br.close();
		
		return featureToIndex;
	}
	
	

	private static int detectClass(double politicsScore, double sportsScore, double techScore) {
		
		if(politicsScore >= sportsScore && politicsScore >= techScore) {
			if(politicsScore > THRESHOLD)
				return POLITICS_LABEL;
			
		}
		else if(sportsScore >= politicsScore && sportsScore >= techScore) {
			if(sportsScore >= THRESHOLD)
				return SPORTS_LABEL;
		}
		else {
			if(techScore > THRESHOLD)
				return TECH_LABEL;
		}
		return GENERAL_LABEL;
	}
	
	private static double evaluateModel(String doc, SVMLightModel svmModel,
			HashMap<String, Integer> featureToIndex) {
		int anyLabel = 0; // not important
		LabeledFeatureVector featureVector = createFeatureVector(doc, featureToIndex, anyLabel);
		if(featureVector == null){
			System.out.println("cannot get feature vector from document");
			System.out.println(doc);
			return 0.0;
		}
		double score = svmModel.classify(featureVector);
		return score;
	}
		
	/*
	private static ArrayList<LabeledFeatureVector> constructTrainingVectors(ArrayList<String> politicsTrainingDocs,
			ArrayList<String> sportsTrainingDocs, ArrayList<String> techTrainingDocs,
			ArrayList<String> stopWords, HashMap<String, Integer> featureToIndex) {

		ArrayList<LabeledFeatureVector> trainingVectors = new ArrayList<>();

		for (String doc : politicsTrainingDocs) {
			LabeledFeatureVector featureVector = createFeatureVector(doc, featureToIndex, POLITICS_LABEL);
			if (featureVector != null) {
				trainingVectors.add(featureVector);
			}
		}

		for (String doc : sportsTrainingDocs) {
			LabeledFeatureVector featureVector = createFeatureVector(doc, featureToIndex, SPORTS_LABEL);
			if (featureVector != null) {
				trainingVectors.add(featureVector);
			}
		}
		
		for (String doc : techTrainingDocs) {
			LabeledFeatureVector featureVector = createFeatureVector(doc, featureToIndex, TECH_LABEL);
			if (featureVector != null) {
				trainingVectors.add(featureVector);
			}
		}

		return trainingVectors;
	}*/
	

	private static LabeledFeatureVector createFeatureVector(String doc, HashMap<String, Integer> featureToIndex,
			int label) {

		String text = doc;
		text = text.trim();
		if (text.length() == 0)
			return null;

		String[] tokens = text.split(" +");
		if (tokens.length == 0)
			return null;

		ArrayList<Integer> documentFeatures = getfeaturesFromDocument(featureToIndex, tokens);
		LabeledFeatureVector labeledFeatureVector = createLabeledFeatureVector(documentFeatures, label);
		return labeledFeatureVector;
	}
	
	private static ArrayList<Integer> getfeaturesFromDocument(HashMap<String, Integer> featuresMap, String[] tokens) {

		ArrayList<Integer> documentFeatures = new ArrayList<Integer>();
		for (String token : tokens) {
			token = token.trim();
			if (token.length() == 0)
				continue;

			Integer index = featuresMap.get(token);
			if (index != null)
				if (!documentFeatures.contains(index))
					documentFeatures.add(index);

		}
		Collections.sort(documentFeatures);
		
		return documentFeatures;
	}

	private static LabeledFeatureVector createLabeledFeatureVector(ArrayList<Integer> documentFeatures, int label) {
		int nDims = documentFeatures.size();
		int[] dims = new int[nDims];
		double[] values = new double[nDims];

		int i = 0;
		for (; i < documentFeatures.size(); i++) {
			dims[i] = documentFeatures.get(i);
			values[i] = 1.0; // could be term frequency or tf-idf
		}

		LabeledFeatureVector labelFeatureVector = new LabeledFeatureVector(label, dims, values);
		labelFeatureVector.normalizeL2();
		return labelFeatureVector;
	}

	private static ArrayList<String> readDocuments(String foldername) throws IOException {
		ArrayList<String> trainingDocs = new ArrayList<>();

		File directory = new File(foldername);

		// get all the files from a directory
		File[] fList = directory.listFiles();
		for (File file : fList) {
			if (file.isFile()) {
				String filePath = foldername + "\\" + file.getName();
				trainingDocs.add(readDocument(filePath));
			}
		}

		return trainingDocs;
	}
	
	private static String readDocument(String fileName) throws IOException {

		String sCurrentLine;
		StringBuilder sb = new StringBuilder("");
		BufferedReader br = new BufferedReader(new FileReader(fileName));

		while ((sCurrentLine = br.readLine()) != null) {
			sb.append(sCurrentLine + " ");
		}
		br.close();

		return sb.toString().replaceAll(NON_ARABIC_CHARACHTER, " ");
	}
	

}

