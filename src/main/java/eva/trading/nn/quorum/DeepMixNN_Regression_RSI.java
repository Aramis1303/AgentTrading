/*
 EVA Prodaction
 */
package eva.trading.nn.quorum;

import eva.trading.Settings;
import eva.trading.data.QDataset;
import eva.trading.data.QState;
import eva.trading.nn.IDeepMixNN_Regression;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Random;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.iterator.IteratorDataSetIterator;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional;
import org.deeplearning4j.nn.conf.preprocessor.CnnToRnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.RnnToCnnPreProcessor;
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
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 *
 * @author Eva
 */
public class DeepMixNN_Regression_RSI implements IDeepMixNN_Regression  {
    private int lengthOfHistory;
    
    private List <DataSet> fullData;
    
    private List <DataSet> trainDataset;
    private List <DataSet> testDataset;
    
    private MultiLayerNetwork networkModel;
    private MultiLayerNetwork dumpModel;
    private MultiLayerConfiguration configOfNet;
    
    private double accuracy = 0;
    private double precision = 0;
    private double recall = 0;
    private double score = 0;
    
    private double mae = 0;
    private double mqe = 0;
    
    private double learningRate = Settings.LEARNING_RATE;
    private double regularization = Settings.REGULARIZATION;
    
    private final int BATCH_TRAINING = Settings.BATCH_TRAINING;
    
    private final int BATCH = 1;
    private final int CHANNELS = 1;
    
    // 12 параметрок на входе превращаем в матрицу 3x4
    private final int TIMESERIES_NUM = 5;
    private final int HYSTORY = Settings.HISTORY_LENGHT;
    
    private final int OUTPUT = 1;
    
    private final int KERNEL_SIZE = 5;
    private final int KERNEL_STEP = 1;
    
    private boolean isPreparedDatasets = false;
    
    // Рандомайзер
    private Random r;
    
    // Создание новой сети
    public DeepMixNN_Regression_RSI (int inputs) {
        this.lengthOfHistory = inputs;
        
        int size_after_conv_t = (TIMESERIES_NUM - KERNEL_SIZE) / KERNEL_STEP +1;
        int size_after_conv_h = (HYSTORY - KERNEL_SIZE) / KERNEL_STEP +1;
        
        configOfNet = new NeuralNetConfiguration.Builder()
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .seed(12345)
            .weightInit(WeightInit.XAVIER)
            .updater(new Adam.Builder().learningRate(learningRate).build())   // AdaDelta, AdaGrad, Adam, AdaMax, AMSGrad, Nadam, Nesterovs, NoOp, RmsProp, Sgd
            .l1(regularization)
            .l2(regularization)
            .list()
            .layer(0, new ConvolutionLayer.Builder(KERNEL_SIZE, KERNEL_SIZE)
                    .name("convL_0")
                    .nIn(CHANNELS)
                    .nOut(OUTPUT)
                    .stride(KERNEL_STEP, KERNEL_STEP)
                    .activation(Activation.SOFTSIGN)
                    .build())
            .layer(1, new LSTM.Builder()
                    .name("lstm_1")
                    .activation(Activation.SOFTSIGN)
                    .nIn(size_after_conv_t * size_after_conv_h * OUTPUT)
                    .nOut(size_after_conv_t * size_after_conv_h * OUTPUT)
                    .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                    .gradientNormalizationThreshold(10)
                    .build())
            .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE)
                    .name("rnn_2")
                    .activation(Activation.IDENTITY)
                    .nIn(size_after_conv_t * size_after_conv_h * OUTPUT)
                    .nOut(1)
                    .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                    .gradientNormalizationThreshold(10)
                    .build())
            //.inputPreProcessor(0, new RnnToCnnPreProcessor(TIMESERIES_NUM, HYSTORY, 1))
            .inputPreProcessor(1, new CnnToRnnPreProcessor(size_after_conv_t, size_after_conv_h, OUTPUT))
            .build();
 
        this.networkModel = new MultiLayerNetwork(configOfNet);
        networkModel.init();
        networkModel.setListeners(new ScoreIterationListener(100));
        
        fullData = new ArrayList<>();
    }
    
    // Загрузка ранее сохранённой сети
    public DeepMixNN_Regression_RSI (int inputs, String path) throws IOException {
        
        Nd4j.getRandom().setSeed(12345);
        
        this.networkModel = ModelSerializer.restoreMultiLayerNetwork(path, true);
        this.lengthOfHistory = inputs;
        
        fullData = new ArrayList<>();
    }
    
    // PUT TRAIN DATA
    public void putData (QDataset qds) {
        
        if (qds.getQStates().size() < 1) throw new RuntimeException("QStates of QDataset less than 1: " + qds.getQStates().size());
        
        // ПЕРЕБОР СОСТОЯНИЙ
        for (int st = 0; st < qds.getQStates().size(); st++) {
            INDArray feturesAll = Nd4j.zeros(1, 1, TIMESERIES_NUM, lengthOfHistory);
            INDArray labelAll   = Nd4j.zeros(1, 1);

            if (qds.getQStates().get(st).getSeries().size() != lengthOfHistory + 1) throw new RuntimeException ("Timeseria length(" + qds.getQStates().get(st).getSeries().size() + ") doesn't equals history length (" + (lengthOfHistory + 1) + ")");

            double allVolume = 0;
            for (int ln = 0; ln < 24; ln++) {
                allVolume += (double)qds.getQStates().get(st).getSeries().get(ln +1).getVolume();
            }
            
            for (int ln = 0; ln < lengthOfHistory; ln++) {
                // ЗАПОЛНЯЕМ ВХОДНЫЕ ДАННЫЕ
                feturesAll.putScalar(0, 0, 0, ln,  (qds.getQStates().get(st).getSeries().get(ln +1).getRsi_3() * 2 / 100) -1);
                feturesAll.putScalar(0, 0, 1, ln,  (qds.getQStates().get(st).getSeries().get(ln +1).getRsi_7() * 2 / 100) -1);
                feturesAll.putScalar(0, 0, 2, ln,  (qds.getQStates().get(st).getSeries().get(ln +1).getRsi_13() * 2 / 100) -1);
                feturesAll.putScalar(0, 0, 3, ln,  (qds.getQStates().get(st).getSeries().get(ln +1).getRsi_21() * 2 / 100) -1);
                feturesAll.putScalar(0, 0, 4, ln,  (qds.getQStates().get(st).getSeries().get(ln +1).getRsi_31() * 2 / 100) -1);
            }

            // ЗАПОЛНЯЕМ ОЖИДАЕМЫЕ ДАННЫЕ
            double close = qds.getQStates().get(qds.getQStates().size() -1).getSeries().get(lengthOfHistory -1).getClose();
            double open = qds.getQStates().get(st).getSeries().get(lengthOfHistory -1).getOpen();
            
            //System.out.println(" \t" + (close / open - 1) * 100);
            
            labelAll.putScalar(0, 0, (close / open - 1) * 100);
            
            fullData.add(new DataSet(feturesAll, labelAll));
        }
        
        //System.out.println("=====================================================================");
        
        isPreparedDatasets = false;
    }
    
    // Устанавливаем связь процесса обучения с web gui
    // http://localhost:9000/train/overview
    public void setWebUI (UIServer uiServer) {
        //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
        StatsStorage statsStorage = new InMemoryStatsStorage();         //Alternative: new FileStatsStorage(File), for saving and loading later
        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
        uiServer.attach(statsStorage);
        //Then add the StatsListener to collect this information from the network, as it trains
        networkModel.setListeners(new StatsListener(statsStorage));
    }
    
    public void prepareDatasets() {
        trainDataset = new ArrayList<>();
        testDataset = new ArrayList<>();
        
        int num;
        r = new Random();
        
        isPreparedDatasets = true;
        fullData = DataSet.merge(fullData).asList();
        
        // Копируем рандомные примеры для теста
        while (testDataset.size() * 100 / fullData.size()  < Settings.TEST_DATA_PERCENT) {
            num = r.nextInt(fullData.size());
            
            testDataset.add(fullData.get(num));
            fullData.remove(num);
        }
        // Копируем данные для обучения
        for (int i = 0; i < fullData.size(); i++) {
            trainDataset.add(fullData.get(i));
        }
        
        System.out.println("Training data size is: \t" + trainDataset.size());
        System.out.println("Testing data size is: \t" + testDataset.size());
        
        // Очищаем данные, чтобы не смешались с новой партией
        fullData.clear();
    }
    
    // Тренировать сеть: ручной режим
    public void trainingNetwork (int epoch) {
        if (!isPreparedDatasets) throw new RuntimeException ("Datasets aren't prepared.");
        
        long start = new Date().getTime();
        
        if (epoch < 1) throw new RuntimeException ("Argument of trainingNetwork can't be less 1");
        
        for(int i = 0; i < epoch; i++ ) {
            IteratorDataSetIterator idsi = new IteratorDataSetIterator(trainDataset.iterator(), BATCH_TRAINING);
            networkModel.fit(idsi);
            score = networkModel.score();
            idsi = null;
        }
        
        System.out.println("trainingNetwork: " + ((new Date().getTime() - start)/1000) + " sec.");
    }
    
    // Протестировать сеть
    public void testNetworkRegression () {
        if (!isPreparedDatasets) throw new RuntimeException ("Datasets aren't prepared.");
        
        RegressionEvaluation eval = networkModel.evaluateRegression(new IteratorDataSetIterator(testDataset.iterator(), BATCH));
        
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
    
    // Использовать обученную сеть 
    // Свеча с индексом 0 должна находится сама старая.
    public double prediction (QState qs) {
        if (qs.getSeries().size() != lengthOfHistory + 1) throw new RuntimeException("It needs "+ (lengthOfHistory + 1) +" candles for prediction.");
        
        INDArray feturesAll = Nd4j.zeros(1, 1, TIMESERIES_NUM, lengthOfHistory);
        INDArray labelAll   = Nd4j.zeros(1, 1);
        
        double allVolume = 0;
        for (int ln = 0; ln < lengthOfHistory; ln++) {
            allVolume += (double)qs.getSeries().get(ln +1).getVolume();
        }
        
        for (int ln = 0; ln < lengthOfHistory; ln++) {
            // ЗАПОЛНЯЕМ ВХОДНЫЕ ДАННЫЕ
            feturesAll.putScalar(0, 0, 0, ln,  (qs.getSeries().get(ln +1).getRsi_3() * 2 / 100) -1);
            feturesAll.putScalar(0, 0, 1, ln,  (qs.getSeries().get(ln +1).getRsi_7() * 2 / 100) -1);
            feturesAll.putScalar(0, 0, 2, ln,  (qs.getSeries().get(ln +1).getRsi_13() * 2 / 100) -1);
            feturesAll.putScalar(0, 0, 3, ln,  (qs.getSeries().get(ln +1).getRsi_21() * 2 / 100) -1);
            feturesAll.putScalar(0, 0, 4, ln,  (qs.getSeries().get(ln +1).getRsi_31() * 2 / 100) -1);
        }
        
        // Собираем данные в датасет
        DataSet predictionData = new DataSet(feturesAll, labelAll);
        
        INDArray output = networkModel.output(predictionData.getFeatures());
        
        return output.getDouble(0);
    }
    
    // Сохранить текущую сеть
    public void saveCurrentNeuralNetwork (String pathBinFile) throws IOException {
        File f = new File(pathBinFile);
        if (f.exists()) {
            f.delete();
        }
        
        ModelSerializer.writeModel(networkModel, pathBinFile, true);
    }
    
    // Сохранить лучшую сеть
    public void saveBestNeuralNetwork (String pathBinFile) throws IOException {
        File f = new File(pathBinFile);
        if (f.exists()) {
            f.delete();
        }
        if (dumpModel != null) {
            ModelSerializer.writeModel(dumpModel, pathBinFile, true); 
        }
        else { 
            System.out.println("Best model doesn't saved 'cause is NULL.");
        }
    }
    
    public void dumpNetwork() {
        dumpModel = networkModel.clone();
    }
    
    public boolean dumpIsNull() {
        if (dumpModel == null) {
            return true;
        }
        else return false;
    }
}
