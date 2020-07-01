import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Random;

public class ClassifierVGG16 {
    private static Logger log = LoggerFactory.getLogger(ClassifierVGG16.class);

    public static void main(String[] args) throws Exception {
        int height = 224;
        int width = 224;
        int channels = 3;
        int batchSize = 32;
        int nEpochs = 5;
        long seed = 990428;
        int numOutput = 2;
        double learningRate = 0.001;
        final String FREEZE_UNTIL_LAYER = "fc2";
        Random randNum = new Random(seed);

        ZooModel zooModel = VGG16.builder().build();
        log.info("VGG16 model is getting download...");
        ComputationGraph preTrainedNet = (ComputationGraph) zooModel.initPretrained(PretrainedType.IMAGENET);

        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .updater(new Adam(learningRate))
                .seed(seed)
                .build();

        ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(preTrainedNet)
                .fineTuneConfiguration(fineTuneConf)
                .setFeatureExtractor(FREEZE_UNTIL_LAYER)
                .removeVertexKeepConnections("predictions")
                .setWorkspaceMode(WorkspaceMode.ENABLED)
                .addLayer("predictions", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(4096)
                        .nOut(numOutput)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.SOFTMAX)
                        .build(), FREEZE_UNTIL_LAYER).build();

        ParentPathLabelGenerator label = new ParentPathLabelGenerator();
        BalancedPathFilter balancedPathFilter = new BalancedPathFilter(randNum, BaseImageLoader.ALLOWED_FORMATS, label);

        File dataset = new File("D:\\dataset\\training_set");
        FileSplit data = new FileSplit(dataset, NativeImageLoader.ALLOWED_FORMATS, randNum);

        InputSplit[] trainvalSplit = data.sample(balancedPathFilter, 80, 20);
        InputSplit train = trainvalSplit[0];
        InputSplit val = trainvalSplit[1];

        ImageRecordReader trainRecordReader = new ImageRecordReader(height, width, channels, label);
        ImageRecordReader valRecordReader = new ImageRecordReader(height, width, channels, label);

        trainRecordReader.initialize(train);
        valRecordReader.initialize(val);

        DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRecordReader, batchSize, 1, numOutput);
        DataSetIterator valIter = new RecordReaderDataSetIterator(valRecordReader, batchSize, 1, numOutput);

        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        trainIter.setPreProcessor(scaler);
        valIter.setPreProcessor(scaler);


        vgg16Transfer.setListeners(new ScoreIterationListener(100));

        for (int i = 0; i < nEpochs; i++) {
            trainIter.reset();
            System.out.println("Epoch " + i);
            vgg16Transfer.fit(trainIter);
            System.out.println("Done an epoch, validating");
            Evaluation evaluation = vgg16Transfer.evaluate(valIter);
            System.out.println(evaluation.stats());
        }

        log.info("***** SAVED MODEL *****");
        File loc = new File("model/vgg16_classifier.zip");
        boolean saveUpdater = true;
        ModelSerializer.writeModel(vgg16Transfer, loc, saveUpdater);

    }
}
