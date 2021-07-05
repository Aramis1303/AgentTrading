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
import org.deeplearning4j.datasets.iterator.IteratorMultiDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 *
 * @author Eva
 */
public class MultiDeepNeuralNetwork_4D {
    private int lengthOfHistory;
    
    private List <MultiDataSet> fullTrainData;
    private List <MultiDataSet> fullTestData;
    
    private List <org.nd4j.linalg.dataset.api.MultiDataSet> preparedTrainDataset;
    private List <org.nd4j.linalg.dataset.api.MultiDataSet> preparedTestDataset;
    
    private ComputationGraph networkModel;
    private ComputationGraphConfiguration configOfNet;
    
    private double accuracy = 0;
    private double precision = 0;
    private double recall = 0;
    private double score = 0;
    
    private double mae = 0;
    private double mqe = 0;
    
    private final double LEARNING_RATE = 0.025;
    private final int BATCH_SIZE = 32;
    
    private EQueryStatuses typeDial;
    
    private boolean isPreparedDatasets = false;
    
    // Создание новой сети
    public MultiDeepNeuralNetwork_4D (int inputs, EQueryStatuses td) {
        this.lengthOfHistory = inputs;
        
        configOfNet = new NeuralNetConfiguration.Builder()
                .graphBuilder()
                
                .addInputs("open", "close", "high", "low", "volume", "ema_f", "ema_s", "stoh_m", "stoh_s", "rsi", "adx", "di_p", "di_m")
                // ВХОДЯЩИЕ СЛОИ
                .addLayer("lstm_open", new LSTM.Builder()
                        .nIn(inputs)
                        .nOut(inputs)
                        .activation(Activation.SOFTSIGN)
                        .l2(0.0003)
                        .updater(new Adam.Builder().learningRate(LEARNING_RATE).build())
                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                        .gradientNormalizationThreshold(10)
                        .build(), "open")
                .addLayer("lstm_close", new LSTM.Builder()
                        .nIn(inputs)
                        .nOut(inputs)
                        .activation(Activation.SOFTSIGN)
                        .l2(0.0003)
                        .updater(new Adam.Builder().learningRate(LEARNING_RATE).build())
                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                        .gradientNormalizationThreshold(10)
                        .build(), "close")
                .addLayer("lstm_high", new LSTM.Builder()
                        .nIn(inputs)
                        .nOut(inputs)
                        .activation(Activation.SOFTSIGN)
                        .l2(0.0003)
                        .updater(new Adam.Builder().learningRate(LEARNING_RATE).build())
                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                        .gradientNormalizationThreshold(10)
                        .build(), "high")
                .addLayer("lstm_low", new LSTM.Builder()
                        .nIn(inputs)
                        .nOut(inputs)
                        .activation(Activation.SOFTSIGN)
                        .l2(0.0003)
                        .updater(new Adam.Builder().learningRate(LEARNING_RATE).build())
                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                        .gradientNormalizationThreshold(10)
                        .build(), "low")
                .addLayer("lstm_volume", new LSTM.Builder()
                        .nIn(inputs)
                        .nOut(inputs)
                        .activation(Activation.SOFTSIGN)
                        .l2(0.0003)
                        .updater(new Adam.Builder().learningRate(LEARNING_RATE).build())
                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                        .gradientNormalizationThreshold(10)
                        .build(), "volume")
                .addLayer("lstm_ema_f", new LSTM.Builder()
                        .nIn(inputs)
                        .nOut(inputs)
                        .activation(Activation.SOFTSIGN)
                        .l2(0.0003)
                        .updater(new Adam.Builder().learningRate(LEARNING_RATE).build())
                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                        .gradientNormalizationThreshold(10)
                        .build(), "ema_f")
                .addLayer("lstm_ema_s", new LSTM.Builder()
                        .nIn(inputs)
                        .nOut(inputs)
                        .activation(Activation.SOFTSIGN)
                        .l2(0.0003)
                        .updater(new Adam.Builder().learningRate(0.001).build())
                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                        .gradientNormalizationThreshold(10)
                        .build(), "ema_s")
                .addLayer("lstm_stoh_m", new LSTM.Builder()
                        .nIn(inputs)
                        .nOut(inputs)
                        .activation(Activation.SOFTSIGN)
                        .l2(0.0003)
                        .updater(new Adam.Builder().learningRate(LEARNING_RATE).build())
                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                        .gradientNormalizationThreshold(10)
                        .build(), "stoh_m")
                .addLayer("lstm_stoh_s", new LSTM.Builder()
                        .nIn(inputs)
                        .nOut(inputs)
                        .activation(Activation.SOFTSIGN)
                        .l2(0.0003)
                        .updater(new Adam.Builder().learningRate(LEARNING_RATE).build())
                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                        .gradientNormalizationThreshold(10)
                        .build(), "stoh_s")
                .addLayer("lstm_rsi", new LSTM.Builder()
                        .nIn(inputs)
                        .nOut(inputs)
                        .activation(Activation.SOFTSIGN)
                        .l2(0.0003)
                        .updater(new Adam.Builder().learningRate(LEARNING_RATE).build())
                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                        .gradientNormalizationThreshold(10)
                        .build(), "rsi")
                .addLayer("lstm_adx", new LSTM.Builder()
                        .nIn(inputs)
                        .nOut(inputs)
                        .activation(Activation.SOFTSIGN)
                        .l2(0.0003)
                        .updater(new Adam.Builder().learningRate(LEARNING_RATE).build())
                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                        .gradientNormalizationThreshold(10)
                        .build(), "adx")
                .addLayer("lstm_di_p", new LSTM.Builder()
                        .nIn(inputs)
                        .nOut(inputs)
                        .activation(Activation.SOFTSIGN)
                        .l2(0.0003)
                        .updater(new Adam.Builder().learningRate(LEARNING_RATE).build())
                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                        .gradientNormalizationThreshold(10)
                        .build(), "di_p")
                .addLayer("lstm_di_m", new LSTM.Builder()
                        .nIn(inputs)
                        .nOut(inputs)
                        .activation(Activation.SOFTSIGN)
                        .l2(0.0003)
                        .updater(new Adam.Builder().learningRate(LEARNING_RATE).build())
                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                        .gradientNormalizationThreshold(10)
                        .build(), "di_m")
                
                .addLayer("deep_layer", new LSTM.Builder()
                        .nIn(inputs * 13)
                        .nOut(inputs)
                        .activation(Activation.SOFTSIGN)
                        .l2(0.0003)
                        .updater(new Adam.Builder().learningRate(LEARNING_RATE).build())
                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                        .gradientNormalizationThreshold(10)
                        .build(), "lstm_open", "lstm_close", "lstm_high", "lstm_low", "lstm_volume", "lstm_ema_f", "lstm_ema_s", "lstm_stoh_m", "lstm_stoh_s", "lstm_rsi", "lstm_adx", "lstm_di_p", "lstm_di_m")
                
                // ВЫХОДНОЙ СЛОЙ
                .addLayer("out", new RnnOutputLayer.Builder(LossFunctions.LossFunction.L2)
                        .nIn(inputs)
                        .nOut(1)
                        .activation(Activation.IDENTITY)
                        .l2(0.0003)
                        .updater(new Adam.Builder().learningRate(0.001).build())
                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                        .gradientNormalizationThreshold(10)
                        .build(), "deep_layer")
                .setOutputs("out")
                .build();
 
        this.networkModel = new ComputationGraph(configOfNet);
        networkModel.init();
        networkModel.setListeners(new ScoreIterationListener(100));
        
        fullTestData = new ArrayList<>();
        fullTrainData = new ArrayList<>();
        
        this.typeDial = td;
    }
    
    // Загрузка ранее сохранённой сети
    public MultiDeepNeuralNetwork_4D (int inputs, String path, EQueryStatuses td) throws IOException {
        
        Nd4j.getRandom().setSeed(12345);
        
        this.networkModel = ModelSerializer.restoreComputationGraph(path, true);
        this.lengthOfHistory = inputs;
        
        fullTestData = new ArrayList<>();
        fullTrainData = new ArrayList<>();
        
        this.typeDial = td;
    }
    
    // PUT TRAIN DATA
    public void putTrainData (QDataset qds) {
        
        if (qds.getQStates().size() < 1) throw new RuntimeException("QStates of QDataset less than 1: " + qds.getQStates().size());
        
        INDArray [] feturesAll = new INDArray [13];
        INDArray [] labelAll = new INDArray [1];
        
        feturesAll[0] = Nd4j.zeros(1, lengthOfHistory, qds.getQStates().size());
        feturesAll[1] = Nd4j.zeros(1, lengthOfHistory, qds.getQStates().size());
        feturesAll[2] = Nd4j.zeros(1, lengthOfHistory, qds.getQStates().size());
        feturesAll[3] = Nd4j.zeros(1, lengthOfHistory, qds.getQStates().size());
        feturesAll[4] = Nd4j.zeros(1, lengthOfHistory, qds.getQStates().size());
        feturesAll[5] = Nd4j.zeros(1, lengthOfHistory, qds.getQStates().size());
        feturesAll[6] = Nd4j.zeros(1, lengthOfHistory, qds.getQStates().size());
        feturesAll[7] = Nd4j.zeros(1, lengthOfHistory, qds.getQStates().size());
        feturesAll[8] = Nd4j.zeros(1, lengthOfHistory, qds.getQStates().size());
        feturesAll[9] = Nd4j.zeros(1, lengthOfHistory, qds.getQStates().size());
        feturesAll[10] = Nd4j.zeros(1, lengthOfHistory, qds.getQStates().size());
        feturesAll[11] = Nd4j.zeros(1, lengthOfHistory, qds.getQStates().size());
        feturesAll[12] = Nd4j.zeros(1, lengthOfHistory, qds.getQStates().size());
        
        labelAll[0] = Nd4j.zeros(1, 1, qds.getQStates().size());
        
        // ПЕРЕБОР СОСТОЯНИЙ
        for (int st = 0; st < qds.getQStates().size(); st++) {
            if (qds.getQStates().get(st).getSeries().size() != lengthOfHistory + 1) throw new RuntimeException ("Timeseria length(" + qds.getQStates().get(st).getSeries().size() + ") doesn't equals history length (" + (lengthOfHistory + 1) + ")");
            
            double allVolume = 0;
            for (int ln = 0; ln < lengthOfHistory; ln++) {
                allVolume += (double)qds.getQStates().get(st).getSeries().get(ln +1).getVolume();
            }
            
            for (int ln = 0; ln < lengthOfHistory; ln++) {
                // ЗАПОЛНЯЕМ ВХОДНЫЕ ДАННЫЕ
                feturesAll[0].putScalar(0, ln, st, (qds.getQStates().get(st).getSeries().get(ln +1).getOpen() / qds.getQStates().get(st).getSeries().get(ln).getClose() -1) * Settings.MAX_TAKE_PERCENT);
                feturesAll[1].putScalar(0, ln, st, (qds.getQStates().get(st).getSeries().get(ln +1).getClose() / qds.getQStates().get(st).getSeries().get(ln).getClose() -1) * Settings.MAX_TAKE_PERCENT);
                feturesAll[2].putScalar(0, ln, st, (qds.getQStates().get(st).getSeries().get(ln +1).getHigh() / qds.getQStates().get(st).getSeries().get(ln).getClose() -1) * Settings.MAX_TAKE_PERCENT);
                feturesAll[3].putScalar(0, ln, st, (qds.getQStates().get(st).getSeries().get(ln +1).getLow() / qds.getQStates().get(st).getSeries().get(ln).getClose() -1) * Settings.MAX_TAKE_PERCENT);
                feturesAll[4].putScalar(0, ln, st, (qds.getQStates().get(st).getSeries().get(ln +1).getVolume() / allVolume * 2) -1);
            }
            
            // ЗАПОЛНЯЕМ ОЖИДАЕМЫЕ ДАННЫЕ
            switch (typeDial) {
                case BUY: {
                    labelAll[0].putScalar(0, 0, st, qds.getQStates().get(st).getRewBuy());
                }; break;
                case SELL: {
                    labelAll[0].putScalar(0, 0, st, qds.getQStates().get(st).getRewSell());
                }; break;
                case WAIT: {
                    labelAll[0].putScalar(0, 0, st, qds.getQStates().get(st).getRewWait());
                }; break;
                case CLOSE: {
                    labelAll[0].putScalar(0, 0, st, qds.getQStates().get(st).getRewClose());
                }; break;
                default: throw new RuntimeException("Type dial for this MDNN is " + typeDial + ". Correct types: BUY, SELL, WAIT, CLOSE");
            }
        }
        
        fullTrainData.add(new MultiDataSet(feturesAll, labelAll));
        isPreparedDatasets = false;
    }
    
    // PUT TEST DATA
    public void putTestData (QDataset qds) {
        
        if (qds.getQStates().size() < 1) throw new RuntimeException("QStates of QDataset less than 1: " + qds.getQStates().size());
        
        INDArray [] feturesAll = new INDArray [13];
        INDArray [] labelAll = new INDArray [1];
        
        feturesAll[0] = Nd4j.zeros(1, lengthOfHistory, qds.getQStates().size());
        feturesAll[1] = Nd4j.zeros(1, lengthOfHistory, qds.getQStates().size());
        feturesAll[2] = Nd4j.zeros(1, lengthOfHistory, qds.getQStates().size());
        feturesAll[3] = Nd4j.zeros(1, lengthOfHistory, qds.getQStates().size());
        feturesAll[4] = Nd4j.zeros(1, lengthOfHistory, qds.getQStates().size());
        feturesAll[5] = Nd4j.zeros(1, lengthOfHistory, qds.getQStates().size());
        feturesAll[6] = Nd4j.zeros(1, lengthOfHistory, qds.getQStates().size());
        feturesAll[7] = Nd4j.zeros(1, lengthOfHistory, qds.getQStates().size());
        feturesAll[8] = Nd4j.zeros(1, lengthOfHistory, qds.getQStates().size());
        feturesAll[9] = Nd4j.zeros(1, lengthOfHistory, qds.getQStates().size());
        feturesAll[10] = Nd4j.zeros(1, lengthOfHistory, qds.getQStates().size());
        feturesAll[11] = Nd4j.zeros(1, lengthOfHistory, qds.getQStates().size());
        feturesAll[12] = Nd4j.zeros(1, lengthOfHistory, qds.getQStates().size());
        
        labelAll[0] = Nd4j.zeros(1, 1, qds.getQStates().size());
        
        // ПЕРЕБОР СОСТОЯНИЙ
        for (int st = 0; st < qds.getQStates().size(); st++) {
            
            if (qds.getQStates().get(st).getSeries().size() != lengthOfHistory + 1) throw new RuntimeException ("Timeseria length(" + qds.getQStates().get(st).getSeries().size() + ") doesn't equals history length (" + (lengthOfHistory + 1) + ")");
            
            double allVolume = 0;
            for (int ln = 0; ln < lengthOfHistory; ln++) {
                allVolume += (double)qds.getQStates().get(st).getSeries().get(ln +1).getVolume();
            }
            
            for (int ln = 0; ln < lengthOfHistory; ln++) {
                // ЗАПОЛНЯЕМ ВХОДНЫЕ ДАННЫЕ
                feturesAll[0].putScalar(0, ln, st, (qds.getQStates().get(st).getSeries().get(ln +1).getOpen() / qds.getQStates().get(st).getSeries().get(ln).getClose() -1) * Settings.MAX_TAKE_PERCENT);
                feturesAll[1].putScalar(0, ln, st, (qds.getQStates().get(st).getSeries().get(ln +1).getClose() / qds.getQStates().get(st).getSeries().get(ln).getClose() -1) * Settings.MAX_TAKE_PERCENT);
                feturesAll[2].putScalar(0, ln, st, (qds.getQStates().get(st).getSeries().get(ln +1).getHigh() / qds.getQStates().get(st).getSeries().get(ln).getClose() -1) * Settings.MAX_TAKE_PERCENT);
                feturesAll[3].putScalar(0, ln, st, (qds.getQStates().get(st).getSeries().get(ln +1).getLow() / qds.getQStates().get(st).getSeries().get(ln).getClose() -1) * Settings.MAX_TAKE_PERCENT);
                feturesAll[4].putScalar(0, ln, st, (qds.getQStates().get(st).getSeries().get(ln +1).getVolume() / allVolume * 2) -1);
            }
            
            // ЗАПОЛНЯЕМ ОЖИДАЕМЫЕ ДАННЫЕ
            switch (typeDial) {
                case BUY: {
                    labelAll[0].putScalar(0, 0, st, qds.getQStates().get(st).getRewBuy());
                }; break;
                case SELL: {
                    labelAll[0].putScalar(0, 0, st, qds.getQStates().get(st).getRewSell());
                }; break;
                case WAIT: {
                    labelAll[0].putScalar(0, 0, st, qds.getQStates().get(st).getRewWait());
                }; break;
                case CLOSE: {
                    labelAll[0].putScalar(0, 0, st, qds.getQStates().get(st).getRewClose());
                }; break;
                default: throw new RuntimeException("Type dial for this MDNN is " + typeDial + ". Correct types: BUY, SELL, WAIT, CLOSE");
            }
        }
        
        fullTestData.add(new MultiDataSet(feturesAll, labelAll));
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
        preparedTrainDataset = MultiDataSet.merge(fullTrainData).asList();
        preparedTestDataset = MultiDataSet.merge(fullTestData).asList();
        
        fullTrainData.clear();
        fullTestData.clear();
    }
    
    // Тренировать сеть: ручной режим
    public void trainingNetwork (int epoch) {
        if (!isPreparedDatasets) throw new RuntimeException ("Datasets aren't prepared.");
        
        long start = new Date().getTime();
        
        if (epoch < 1) throw new RuntimeException ("Argument of trainingNetwork can't be less 1");
        
        for(int i = 0; i < epoch; i++ ) {
            networkModel.fit(new IteratorMultiDataSetIterator(preparedTrainDataset.iterator(), BATCH_SIZE));
            score = networkModel.score();
        }
        
        System.out.println("trainingNetwork: " + ((new Date().getTime() - start)/1000) + " sec.");
    }
    
    // Протестировать сеть
    public void testNetworkClassification () {
        if (!isPreparedDatasets) throw new RuntimeException ("Datasets aren't prepared.");
        
        Evaluation eval = networkModel.evaluate(new IteratorMultiDataSetIterator(preparedTestDataset.iterator(), BATCH_SIZE));
        
        accuracy = eval.accuracy();
        precision = eval.precision();
        recall = eval.recall();
        
        System.out.println("stats: \n" + eval.stats()); 
    }
    
    // Протестировать сеть
    public void testNetworkRegression () {
        if (!isPreparedDatasets) throw new RuntimeException ("Datasets aren't prepared.");
        
        RegressionEvaluation eval = networkModel.evaluateRegression(new IteratorMultiDataSetIterator(preparedTestDataset.iterator(), 1));
        
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
        
        INDArray [] feturesAll = new INDArray [13];
        INDArray [] labelAll = new INDArray [1];
        
        feturesAll[0] = Nd4j.zeros(1, lengthOfHistory, 1);
        feturesAll[1] = Nd4j.zeros(1, lengthOfHistory, 1);
        feturesAll[2] = Nd4j.zeros(1, lengthOfHistory, 1);
        feturesAll[3] = Nd4j.zeros(1, lengthOfHistory, 1);
        feturesAll[4] = Nd4j.zeros(1, lengthOfHistory, 1);
        feturesAll[5] = Nd4j.zeros(1, lengthOfHistory, 1);
        feturesAll[6] = Nd4j.zeros(1, lengthOfHistory, 1);
        feturesAll[7] = Nd4j.zeros(1, lengthOfHistory, 1);
        feturesAll[8] = Nd4j.zeros(1, lengthOfHistory, 1);
        feturesAll[9] = Nd4j.zeros(1, lengthOfHistory, 1);
        feturesAll[10] = Nd4j.zeros(1, lengthOfHistory, 1);
        feturesAll[11] = Nd4j.zeros(1, lengthOfHistory, 1);
        feturesAll[12] = Nd4j.zeros(1, lengthOfHistory, 1);
        
        labelAll[0] = Nd4j.zeros(1, 1, 1);
        
        double allVolume = 0;
        for (int ln = 0; ln < lengthOfHistory; ln++) {
            allVolume += (double)qs.getSeries().get(ln +1).getVolume();
        }
        
        for (int ln = 0; ln < lengthOfHistory; ln++) {
            // ЗАПОЛНЯЕМ ВХОДНЫЕ ДАННЫЕ
            feturesAll[0].putScalar(0, ln, 0, (qs.getSeries().get(ln +1).getOpen() / qs.getSeries().get(ln).getClose() -1) * Settings.MAX_TAKE_PERCENT);
            feturesAll[1].putScalar(0, ln, 0, (qs.getSeries().get(ln +1).getClose() / qs.getSeries().get(ln).getClose() -1) * Settings.MAX_TAKE_PERCENT);
            feturesAll[2].putScalar(0, ln, 0, (qs.getSeries().get(ln +1).getHigh() / qs.getSeries().get(ln).getClose() -1) * Settings.MAX_TAKE_PERCENT);
            feturesAll[3].putScalar(0, ln, 0, (qs.getSeries().get(ln +1).getLow() / qs.getSeries().get(ln).getClose() -1) * Settings.MAX_TAKE_PERCENT);
            feturesAll[4].putScalar(0, ln, 0, (qs.getSeries().get(ln +1).getVolume() / allVolume * 2) -1);
        }
        
        // Собираем данные в датасет
        MultiDataSet predictionData = new MultiDataSet(feturesAll, labelAll);
        
        INDArray[] output = networkModel.output(predictionData.getFeatures());
        
        return output[0].getRow(0).getColumn(0).getDouble(0);
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
