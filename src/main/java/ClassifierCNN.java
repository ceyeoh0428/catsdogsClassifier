import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.File;
import java.util.Random;


public class ClassifierCNN {
    private static Logger log = LoggerFactory.getLogger(ClassifierCNN.class);

    public static void main(String[] args) throws Exception {
        int height = 128;
        int width = 128;
        int channels = 3;
        int seed = 428;
        Random randNum = new Random(seed);
        int batchSize = 32;
        int nEpochs = 10;
        double learningRate = 0.001;
        double momentum = 0.9;
        double reg = 0.0005;
        int numHidden1 = 25;
        int numHidden2 = 50;
        int numOutput = 2;

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


        log.info("***** BUILD MODEL *****");

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .l2(reg)
                .updater(new Nesterovs(learningRate, momentum))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, new ConvolutionLayer.Builder(32, 32)
                        .nIn(channels)
                        .stride(1, 1)
                        .nOut(numHidden1)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(2, new DenseLayer.Builder()
                        .activation(Activation.RELU)
                        .nOut(numHidden2)
                        .build())
                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numOutput)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(height, width, channels))
                .backpropType(BackpropType.Standard)
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        model.setListeners(new ScoreIterationListener(100));

        for (int i = 0; i < nEpochs; i++) {
            trainIter.reset();
            System.out.println("Epoch " + i);
            model.fit(trainIter); //model.fit
            System.out.println("Done an epoch, validating");
            Evaluation evaluation = model.evaluate(valIter);
            System.out.println(evaluation.stats());
        }

        log.info("***** SAVED MODEL *****");
        File loc = new File("cnn_classifier.zip");
        boolean saveUpdater = true;
        ModelSerializer.writeModel(model, loc, saveUpdater);

        log.info("***** SUMMARY OF MODEL *****");
        log.info(model.summary());
    }

}
