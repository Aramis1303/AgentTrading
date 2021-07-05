/*
 EVA Prodaction
 */
package eva.trading.nn;

import eva.trading.Settings;
import eva.trading.data.EDealStates;
import eva.trading.data.EQueryStatuses;
import eva.trading.data.QDataset;
import eva.trading.data.QState;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.iterator.IteratorDataSetIterator;
import org.deeplearning4j.datasets.iterator.IteratorMultiDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.preprocessor.CnnToRnnPreProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 *
 * @author Eva
 */
public class DeepMixNN {
    private int lengthOfHistory;
    
    private List <DataSet> fullTrainData;
    private List <DataSet> fullTestData;
    
    private List <DataSet> preparedTrainDataset;
    private List <DataSet> preparedTestDataset;
    
    private MultiLayerNetwork networkModel;
    private MultiLayerConfiguration configOfNet;
    
    private double accuracy = 0;
    private double precision = 0;
    private double recall = 0;
    private double score = 0;
    
    private double mae = 0;
    private double mqe = 0;
    
    private final int BATCH_TRAINING = 32;
    
    private final int BATCH = 1;
    private final int FILTERS = 1;
    
    // 12 параметрок на входе превращаем в матрицу 3x4
    private final int TIMESERIES_NUM = 13; 
    private final int HYSTORY = 36;
    
    private final int OUTPUT = 5;
    
    private final int KERNEL_SIZE = 3;
    private final int KERNEL_STEP = 1;
    
    private EQueryStatuses typeDial;
    
    private boolean isPreparedDatasets = false;
    
    // Создание новой сети
    public DeepMixNN (int inputs, EQueryStatuses td) {
        this.lengthOfHistory = inputs;
        
        configOfNet = new NeuralNetConfiguration.Builder()
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .seed(12345)
            .weightInit(WeightInit.XAVIER)
            .updater(Updater.ADAGRAD)
            .list()
            .layer(0, new ConvolutionLayer.Builder(KERNEL_SIZE, KERNEL_SIZE)
                    .nIn(FILTERS)
                    .nOut(OUTPUT)
                    .stride(KERNEL_STEP, KERNEL_STEP)
                    .activation(Activation.TANH)
                    .build())
            .layer(1, new LSTM.Builder()
                        .activation(Activation.SOFTSIGN)
                        .nIn(OUTPUT * (TIMESERIES_NUM -2) * (HYSTORY -2))
                        .nOut(HYSTORY * OUTPUT)
                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                        .gradientNormalizationThreshold(10)
                        .build())
            .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE)
                    .activation(Activation.IDENTITY)
                    .nIn(HYSTORY * OUTPUT)
                    .nOut(1)
                    .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                    .gradientNormalizationThreshold(10)
                    .build())
            //.inputPreProcessor(0, new RnnToCnnPreProcessor(PARAMS_H, PARAMS_W, 1))
            .inputPreProcessor(1, new CnnToRnnPreProcessor(TIMESERIES_NUM -2, HYSTORY -2, OUTPUT))
            .build();
 
        this.networkModel = new MultiLayerNetwork(configOfNet);
        networkModel.init();
        networkModel.setListeners(new ScoreIterationListener(100));
        
        fullTestData = new ArrayList<>();
        fullTrainData = new ArrayList<>();
        
        this.typeDial = td;
    }
    
    // Загрузка ранее сохранённой сети
    public DeepMixNN (int inputs, String path, EQueryStatuses td) throws IOException {
        
        Nd4j.getRandom().setSeed(12345);
        
        this.networkModel = ModelSerializer.restoreMultiLayerNetwork(path, true);
        this.lengthOfHistory = inputs;
        
        fullTestData = new ArrayList<>();
        fullTrainData = new ArrayList<>();
        
        this.typeDial = td;
    }
    
    // PUT TRAIN DATA
    public void putTrainData (QDataset qds) {
        
        if (qds.getQStates().size() < 1) throw new RuntimeException("QStates of QDataset less than 1: " + qds.getQStates().size());
        
        // ПЕРЕБОР СОСТОЯНИЙ
        for (int st = 0; st < qds.getQStates().size(); st++) {
            INDArray feturesAll = Nd4j.zeros(1, 1, 5, lengthOfHistory);
            INDArray labelAll   = Nd4j.zeros(1);
            
            if (qds.getQStates().get(st).getSeries().size() != lengthOfHistory + 1) throw new RuntimeException ("Timeseria length(" + qds.getQStates().get(st).getSeries().size() + ") doesn't equals history length (" + (lengthOfHistory + 1) + ")");
            
            double allVolume = 0;
            for (int ln = 0; ln < lengthOfHistory; ln++) {
                allVolume += (double)qds.getQStates().get(st).getSeries().get(ln +1).getVolume();
            }
            
            for (int ln = 0; ln < lengthOfHistory; ln++) {
                // ЗАПОЛНЯЕМ ВХОДНЫЕ ДАННЫЕ
                feturesAll.putScalar(0, 0, 0, ln,  (qds.getQStates().get(st).getSeries().get(ln +1).getOpen() / qds.getQStates().get(st).getSeries().get(ln).getClose() -1) * Settings.MAX_TAKE_PERCENT);
                feturesAll.putScalar(0, 0, 1, ln,  (qds.getQStates().get(st).getSeries().get(ln +1).getClose() / qds.getQStates().get(st).getSeries().get(ln).getClose() -1) * Settings.MAX_TAKE_PERCENT);
                feturesAll.putScalar(0, 0, 2, ln,  (qds.getQStates().get(st).getSeries().get(ln +1).getHigh() / qds.getQStates().get(st).getSeries().get(ln).getClose() -1) * Settings.MAX_TAKE_PERCENT);
                feturesAll.putScalar(0, 0, 3, ln,  (qds.getQStates().get(st).getSeries().get(ln +1).getLow() / qds.getQStates().get(st).getSeries().get(ln).getClose() -1) * Settings.MAX_TAKE_PERCENT);
                feturesAll.putScalar(0, 0, 4, ln,  (qds.getQStates().get(st).getSeries().get(ln +1).getVolume() / allVolume * 2) -1);
            }
            
            // ЗАПОЛНЯЕМ ОЖИДАЕМЫЕ ДАННЫЕ
            switch (typeDial) {
                case BUY: {
                    labelAll.putScalar(0, qds.getQStates().get(st).getRewBuy());
                }; break;
                case SELL: {
                    labelAll.putScalar(0, qds.getQStates().get(st).getRewSell());
                }; break;
                case WAIT: {
                    labelAll.putScalar(0, qds.getQStates().get(st).getRewWait());
                }; break;
                case CLOSE: {
                    labelAll.putScalar(0, qds.getQStates().get(st).getRewClose());
                }; break;
                default: throw new RuntimeException("Type dial for this MDNN is " + typeDial + ". Correct types: BUY, SELL, WAIT, CLOSE");
            }
            fullTrainData.add(new DataSet(feturesAll, labelAll));
            
        }
        isPreparedDatasets = false;
    }
    
    // PUT TEST DATA
    public void putTestData (QDataset qds) {
        
        if (qds.getQStates().size() < 1) throw new RuntimeException("QStates of QDataset less than 1: " + qds.getQStates().size());
        
        // ПЕРЕБОР СОСТОЯНИЙ
        for (int st = 0; st < qds.getQStates().size(); st++) {
            INDArray feturesAll = Nd4j.zeros(1, 1, 5, lengthOfHistory);
            INDArray labelAll   = Nd4j.zeros(1);
            
            if (qds.getQStates().get(st).getSeries().size() != lengthOfHistory + 1) throw new RuntimeException ("Timeseria length(" + qds.getQStates().get(st).getSeries().size() + ") doesn't equals history length (" + (lengthOfHistory + 1) + ")");
            
            double allVolume = 0;
            for (int ln = 0; ln < lengthOfHistory; ln++) {
                allVolume += (double)qds.getQStates().get(st).getSeries().get(ln +1).getVolume();
            }
            
            for (int ln = 0; ln < lengthOfHistory; ln++) {
                // ЗАПОЛНЯЕМ ВХОДНЫЕ ДАННЫЕ
                feturesAll.putScalar(0, 0, 0, ln,  (qds.getQStates().get(st).getSeries().get(ln +1).getOpen() / qds.getQStates().get(st).getSeries().get(ln).getClose() -1) * Settings.MAX_TAKE_PERCENT);
                feturesAll.putScalar(0, 0, 1, ln,  (qds.getQStates().get(st).getSeries().get(ln +1).getClose() / qds.getQStates().get(st).getSeries().get(ln).getClose() -1) * Settings.MAX_TAKE_PERCENT);
                feturesAll.putScalar(0, 0, 2, ln,  (qds.getQStates().get(st).getSeries().get(ln +1).getHigh() / qds.getQStates().get(st).getSeries().get(ln).getClose() -1) * Settings.MAX_TAKE_PERCENT);
                feturesAll.putScalar(0, 0, 3, ln,  (qds.getQStates().get(st).getSeries().get(ln +1).getLow() / qds.getQStates().get(st).getSeries().get(ln).getClose() -1) * Settings.MAX_TAKE_PERCENT);
                feturesAll.putScalar(0, 0, 4, ln,  (qds.getQStates().get(st).getSeries().get(ln +1).getVolume() / allVolume * 2) -1);
            }
            
            // ЗАПОЛНЯЕМ ОЖИДАЕМЫЕ ДАННЫЕ
            switch (typeDial) {
                case BUY: {
                    labelAll.putScalar(0, qds.getQStates().get(st).getRewBuy());
                }; break;
                case SELL: {
                    labelAll.putScalar(0, qds.getQStates().get(st).getRewSell());
                }; break;
                case WAIT: {
                    labelAll.putScalar(0, qds.getQStates().get(st).getRewWait());
                }; break;
                case CLOSE: {
                    labelAll.putScalar(0, qds.getQStates().get(st).getRewClose());
                }; break;
                default: throw new RuntimeException("Type dial for this MDNN is " + typeDial + ". Correct types: BUY, SELL, WAIT, CLOSE");
            }
            fullTestData.add(new DataSet(feturesAll, labelAll));
            
        }
        isPreparedDatasets = false;
    }
    
    // Устанавливаем связь процесса обучения с web gui
    public void setWebUI (UIServer uiServer) {
        //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
        StatsStorage statsStorage = new InMemoryStatsStorage();         //Alternative: new FileStatsStorage(File), for saving and loading later
        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
        uiServer.attach(statsStorage);
        //Then add the StatsListener to collect this information from the network, as it trains
        networkModel.setListeners(new StatsListener(statsStorage));
    }
    
    public void prepareDatasets() {
        isPreparedDatasets = true;
        preparedTrainDataset = DataSet.merge(fullTrainData).asList();
        preparedTestDataset = DataSet.merge(fullTestData).asList();
        
        fullTrainData.clear();
        fullTestData.clear();
    }
    
    // Тренировать сеть: ручной режим
    public void trainingNetwork (int epoch) {
        if (!isPreparedDatasets) throw new RuntimeException ("Datasets aren't prepared.");
        
        long start = new Date().getTime();
        
        if (epoch < 1) throw new RuntimeException ("Argument of trainingNetwork can't be less 1");
        
        
        
        for(int i = 0; i < epoch; i++ ) {
            IteratorDataSetIterator idsi = new IteratorDataSetIterator(preparedTrainDataset.iterator(), BATCH_TRAINING);
            networkModel.fit(idsi);
            score = networkModel.score();
            idsi = null;
        }
        
        System.out.println("trainingNetwork: " + ((new Date().getTime() - start)/1000) + " sec.");
    }
    
    // Протестировать сеть
    public void testNetworkClassification () {
        if (!isPreparedDatasets) throw new RuntimeException ("Datasets aren't prepared.");
        
        IteratorDataSetIterator idsi = new IteratorDataSetIterator(preparedTestDataset.iterator(), BATCH_TRAINING);
        
        Evaluation eval = networkModel.evaluate(idsi);
        
        accuracy = eval.accuracy();
        precision = eval.precision();
        recall = eval.recall();
        
        eval = null;
        idsi = null;
        
        System.out.println("stats: \n" + eval.stats()); 
    }
    
    // Протестировать сеть
    public void testNetworkRegression () {
        if (!isPreparedDatasets) throw new RuntimeException ("Datasets aren't prepared.");
        
        RegressionEvaluation eval = networkModel.evaluateRegression(new IteratorDataSetIterator(preparedTestDataset.iterator(), 1));
        
        mae = eval.averageMeanAbsoluteError();
        mqe = eval.averageMeanSquaredError();
    }
    
    // Текущая оценка
    public double score() {
        return score;
    }
    public double accuracy() {
        return accuracy;
    }
    public double precision() {
        return precision;
    }
    public double recall() {
        return recall;
    }
    
    public double mae() {
        return mae;
    }
    public double mqe() {
        return mqe;
    }
    
    
    public boolean trainDataIsEmpty() {
        return fullTrainData.isEmpty();
    }
    public boolean testDataIsEmpty() {
        return fullTestData.isEmpty();
    }
    
    
    // Использовать обученную сеть 
    // Свеча с индексом 0 должна находится сама старая.
    public double prediction (QState qs, EDealStates eds) {
        if (qs.getSeries().size() != lengthOfHistory + 1) throw new RuntimeException("It needs "+ (lengthOfHistory + 1) +" candles for prediction.");
        
        INDArray feturesAll = Nd4j.zeros(1, 1, 5, lengthOfHistory);
        INDArray labelAll   = Nd4j.zeros(1);
        
        double allVolume = 0;
        for (int ln = 0; ln < lengthOfHistory; ln++) {
            allVolume += (double)qs.getSeries().get(ln +1).getVolume();
        }
        
        for (int ln = 0; ln < lengthOfHistory; ln++) {
            // ЗАПОЛНЯЕМ ВХОДНЫЕ ДАННЫЕ
            feturesAll.putScalar(0, 0, 0, ln, (qs.getSeries().get(ln +1).getOpen() / qs.getSeries().get(ln).getClose() -1) * Settings.MAX_TAKE_PERCENT);
            feturesAll.putScalar(0, 0, 1, ln, (qs.getSeries().get(ln +1).getClose() / qs.getSeries().get(ln).getClose() -1) * Settings.MAX_TAKE_PERCENT);
            feturesAll.putScalar(0, 0, 2, ln, (qs.getSeries().get(ln +1).getHigh() / qs.getSeries().get(ln).getClose() -1) * Settings.MAX_TAKE_PERCENT);
            feturesAll.putScalar(0, 0, 3, ln, (qs.getSeries().get(ln +1).getLow() / qs.getSeries().get(ln).getClose() -1) * Settings.MAX_TAKE_PERCENT);
            feturesAll.putScalar(0, 0, 4, ln, (qs.getSeries().get(ln +1).getVolume() / allVolume * 2) -1);
        }
        
        // Собираем данные в датасет
        DataSet predictionData = new DataSet(feturesAll, labelAll);
        
        INDArray output = networkModel.output(predictionData.getFeatures());
        
        return output.getRow(0).getColumn(0).getDouble(0);
    }
    
    // Сохранить сеть
    public void saveNeuralNetwork (String pathBinFile) throws IOException {
        File f = new File(pathBinFile);
        if (f.exists()) {
            f.delete();
        }
        ModelSerializer.writeModel(networkModel, pathBinFile, true);
    }
    
    //
    public EQueryStatuses getTypeNN () {
        return typeDial;
    }
    
}
