/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package eva.trading.agent;

import eva.trading.data.QState;
import eva.trading.data.QDataPiece;
import eva.trading.data.QDataset;
import eva.trading.data.EDealStates;
import eva.trading.data.EQueryStatuses;
import eva.trading.Settings;
//import eva.trading.nn.DeepMixNN_Regression_All;
import eva.trading.nn.DeepLSTM_Regression_All;
import eva.trading.sql.SQLQuery;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Collections;
import java.util.GregorianCalendar;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
//import org.sikuli.script.FindFailed;

//import org.sikuli.script.Screen;

/**
 *
 * @author EVA
 */
public class AgentR implements Runnable {
    
    private List <QDataset> datasets = null; // Набор сделок
    
    //private DeepMixNN_Regression_All nn_r = null; // Нейронная сеть агента
    private DeepLSTM_Regression_All nn_r = null; 
    private SQLQuery sql;
    
    private boolean isStoped = false;
    
    private String symbol;
    // Счетчик сделок
    private int counterDeals = 0;
    // Счетчик простоя
    private int downtime_counter = 0;
    
    // Показатели нейронной сети
    private double forecast = 0;
    
    // Календарь
    private Calendar gs;
    
    private Map <Long, Long> buy_test;
    private Map <Long, Long> sell_test;
    
    private double forecast_old = 0;
    
    public AgentR (String symbol_tab) throws IOException {
        
        symbol = symbol_tab;
        
        datasets    = new ArrayList <>();
        
        buy_test = new HashMap<>();
        sell_test = new HashMap<>();
        
        // Создаем подключение к SQL
        sql = new SQLQuery();
        
        gs = new GregorianCalendar();
        
        // Запускаем поток агента
        new Thread(this).start();
    }
    
    //////////////////////////////////////////////////////////////////////////// RUN
    @Override
    public void run() {
        long KEY = 0;
        
        System.out.println(new Date() + ". Preparation...");
        
        if (Settings.IS_TEST_DS_LAUNCH) {
            compileDatasets();
            collect_test_ds();
        }
        
        if (Settings.FIRST_LEARNING_LAUNCH) {
            // Удаляем старую сеть, чтобы не произошоло конфликтов со конфигурациями старой сети
            String [] types = new String[]{"", "core", "ema", "rsi", "stoh", "dx"};
            for (int i = 0; i < types.length; i++) {
                File f = new File(Settings.PATH_NEURON_NET + symbol + types[i]);
                if (f.exists()) {
                    f.delete();
                    System.out.println(Settings.PATH_NEURON_NET + symbol + types[i] + " was deleted.");
                }
            }
            
            compileDatasets();
            try {
                toLearn();
            } catch (IOException ex) {
                Logger.getLogger(AgentR.class.getName()).log(Level.SEVERE, null, ex);
                return;
            }
        }
        else {
            try {
                nn_r = loadMNN();
            } catch (IOException ex) {
                Logger.getLogger(AgentR.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        
        // Цикл работы агента
        long newCheck = new Date().getTime();
        System.out.println(new Date() + ". Trading system is started.");
        while (!isStoped) {
            
            if (newCheck <= new Date().getTime()) {
                newCheck += Settings.AGENT_CYCLE_PERIOD_SEC;
                // Если время последнего запроса не совпадает с последним запросом от MT5 => Пришёл новый запрос
                List <Long> queries = sql.getTimestampsByState(symbol, EQueryStatuses.QUERY);
                // Расчитываем прогноз и пишем его в БД
                for (long timestamp: queries) {
                    System.out.println("TIMESTAMP: \t" + timestamp);
                    forecast = 0.0;
                    
                    QState qs = sql.getQState(symbol, timestamp, Settings.HISTORY_LENGHT + 1);
                    
                    EDealStates eds = sql.getStateByTime(symbol, timestamp);
                    EQueryStatuses eqs = null;
                    
                    if (!Settings.IS_TEST_DS_LAUNCH) {
                        // WORKING
                        forecast = nn_r.prediction(qs);

                        System.out.print("FORECASTS: ");
                        System.out.printf("%.4f, ", forecast);
                        System.out.println();
                        
                        switch (eds) {
                            // Если мы находимся в сделке ЛОНГ
                            case LONG: {
                                if (forecast < 0) {
                                    eqs = EQueryStatuses.CLOSE;
                                }
                                else {
                                    eqs = EQueryStatuses.WAIT;
                                }
                            } break;
                            // Если мы находимся в сделке ЛОНГ
                            case SHORT: {
                                if (forecast > 0) {
                                    eqs = EQueryStatuses.CLOSE;
                                }
                                else {
                                    eqs = EQueryStatuses.WAIT;
                                }
                            } break;
                            // ЕСЛИ У НАС НЕТ СДЕЛКИ
                            default: {
                                if      (forecast > 5) eqs = EQueryStatuses.BUY;
                                else if (forecast < -5) eqs = EQueryStatuses.SELL;
                                else eqs = EQueryStatuses.WAIT;
                            }
                        }
                    }
                    else {
                        //TESTING DATASETS
                        switch (eds) { 
                            // Если мы находимся в сделке ЛОНГ
                            case LONG: {
                                if (buy_test.get(KEY) == timestamp) {
                                    eqs = EQueryStatuses.CLOSE;
                                }
                                else {
                                    eqs = EQueryStatuses.WAIT;
                                }
                            } break;
                            // Если мы находимся в сделке ШОРТ
                            case SHORT: {
                                if (sell_test.get(KEY) == timestamp) {
                                    eqs = EQueryStatuses.CLOSE;
                                }
                                else {
                                    eqs = EQueryStatuses.WAIT;
                                }
                            } break;
                            // ЕСЛИ У НАС НЕТ СДЕЛКИ
                            default: {
                                if (buy_test.containsKey(timestamp))  {
                                    eqs = EQueryStatuses.BUY;
                                    KEY = timestamp;
                                    System.out.println("\nBUY: " + timestamp + " till " + buy_test.get(KEY) + "\n");
                                }
                                else if (sell_test.containsKey(timestamp)) {
                                    eqs = EQueryStatuses.SELL;
                                    KEY = timestamp;
                                    System.out.println("\nSELL: " + timestamp + " till " + sell_test.get(KEY) + "\n");
                                }
                                else eqs = EQueryStatuses.WAIT;
                            }
                        }
                    }
                    
                    sql.setStatus(symbol, eqs, timestamp);
                    if (forecast != 0) sql.setForecast(symbol, forecast, timestamp);
                    
                    forecast_old = forecast;
                }
            }
            
            try {
                Thread.sleep(100); //
            } catch (InterruptedException ex) {
                Logger.getLogger(AgentR.class.getName()).log(Level.SEVERE, null, ex);
            }
        } 
    }
    
    //////////////////////////////////////////////////////////////////////////// ОБУЧЕНИЕ
    public void toLearn() throws IOException {
        
        // Получаем новую сеть
        nn_r = loadMNN();
        
        for (int i = datasets.size() -1; i >= 0; i--) {
            if (datasets.get(i).getQStates().size() < Settings.MIN_DATASET_SIZE || datasets.get(i).getQStates().size() > Settings.MAX_DATASET_SIZE) {
                datasets.remove(i);
            }
        }
        
        // Удаляем самые старые Dataset'ы пока не останется заданное кол-во
        while (datasets.size() > Settings.MAX_QUANTITY_DATASETS) {
            datasets.remove(0);
        }

        // Сортируем датасеты
        Collections.sort(datasets);
        //for(QDataset qds: datasets) {
        //    System.out.println(qds.getReward());
        //}

        // Убираем выбросы в размере 10%
        int outlier = datasets.size() / 10;
        for (int e = 0; e < outlier; e++) {
            if (e%2 == 0) {
                datasets.remove(datasets.size() -1);
            }
            else datasets.remove(0);
        }

        // Собираем историю в наборы данных для обучения
        System.out.println("DATASET SIZE: " + datasets.size());
        if (datasets.size() > 1) {
            System.out.println("DATASET IS LOADING TO NEURAL NETWORKS...");
            for (int d = 0; d < datasets.size(); d++) {
                datasets.get(d).calculatEval();

                nn_r.putData(datasets.get(d));
            }

            // Подготовка данных (разбивка на батчи, перевод в датасеты)
            System.out.println("NEURAL NETWORKS IS PREPARING DATASETS...");
            
            nn_r.prepareDatasets();
            
            // ОБУЧЕНИЕ
            System.out.println("NEURAL NETWORKS IS TRAINING...");
            newThreadLearnProccess(nn_r);

            System.out.println("TRAINING COMPLETED.");
        }
        
        // Загружаем лучший вариант сети
        nn_r = loadMNN();
    }
    
    //////////////////////////////////////////////////////////////////////////// Learn thread
    public void newThreadLearnProccess(DeepLSTM_Regression_All mdnn) throws IOException {
        
        // WEB Графика процесса обучения
        mdnn.setWebUI(UIServer.getInstance());          // http://localhost:9000/train/overview
        
        // Обучаем новую сеть
        double mae = Double.MAX_VALUE;
        boolean access = false;
        for (int f = 0; f < Settings.CYCLE_FIT_NEURON_NET; f++) {
            System.out.println((f+1) + " iteration:");
            // ТЕСТ И СБОР ОЦЕНОК ПОСЛЕ ОБУЧЕНИЯ
            mdnn.trainingNetwork(1);
            mdnn.testNetworkRegression();
            if (mdnn.score() < Settings.PERCENT_TAKE_PROFIT * 1.5 && f > 50) access = true;
            
            if (mae > mdnn.mae() && access) { //&& mdnn.score() < Settings.PERCENT_TAKE_PROFIT) {
                mae = mdnn.mae();
                
                mdnn.dumpNetwork();
                System.out.println("NEURAL NETWORK DUMPED. \tMAE: " + mae);
            }
            else {
                System.out.println("CURRENT MAE: " + mdnn.mae() + ". \tBEST MAE: " + mae);
            }
            
            if (f % 25 == 0 && f != 0) {
                // Сохраняем промежуточный результат обучения
                try {
                    mdnn.saveBestNeuralNetwork(Settings.PATH_NEURON_NET + symbol);
                    System.out.println("NEURAL NETWORK SAVED TO: " + Settings.PATH_NEURON_NET + symbol + ". \t MAE: " + mae);
                } catch (IOException ex) {
                    Logger.getLogger(AgentR.class.getName()).log(Level.SEVERE, null, ex);
                }
            }
        }
        
        // Сохраняем конечный результат обучения
        try {
            mdnn.saveBestNeuralNetwork(Settings.PATH_NEURON_NET + symbol);
            System.out.println("NEURAL NETWORK SAVED TO: " + Settings.PATH_NEURON_NET + symbol + ". \t MAE: " + mae);
        } catch (IOException ex) {
            Logger.getLogger(AgentR.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    //////////////////////////////////////////////////////////////////////////// LOAD MNN
    // Загрузка нейронной сети
    private DeepLSTM_Regression_All loadMNN() throws IOException {
        DeepLSTM_Regression_All nn = null;
        
        if (new File(Settings.PATH_NEURON_NET + symbol).exists()) {
            nn = new DeepLSTM_Regression_All(Settings.HISTORY_LENGHT, Settings.PATH_NEURON_NET + symbol);
            System.out.println("Neural Network was read from file: \"" + Settings.PATH_NEURON_NET + symbol + "\"");
        }
        else {
            nn = new DeepLSTM_Regression_All (Settings.HISTORY_LENGHT);
        }
        File f = new File(Settings.PATH_NEURON_NET);
        try{
            if(f.mkdir()) { 
                System.out.println("Directory \"" + Settings.PATH_NEURON_NET + "\" was created.");
            } 
        } catch(Exception e){
            e.printStackTrace();
        } 
        
        return nn;
    }
    
    //////////////////////////////////////////////////////////////////////////// compileDatasets
    // Создаём датасеты по алгоритму перепадов ценовых показателей
    private void compileDatasets() {
        datasets.clear();
        
        EDealStates esd = EDealStates.NODEAL;
                
        List <QDataPiece> listQDP = sql.getAllHistory(symbol);
        
        //System.out.println("Hystory size is: " + listQDP.size() + " candles.");
        
        // Цена открытия
        double oPrice = 0;
        // Цена закрытия
        double cPrice = 0;
        
        // Точка открытия
        int oPoint = 0;
        // Точка закрытия
        int cPoint = 0;
        
        QDataset qds;
        double eval;
        
        // Сборка dataset
        for (int a = Settings.HISTORY_LENGHT +1; a < listQDP.size(); a++) {
            switch (esd) {
                case NODEAL: {
                    //System.out.println(a + ": NODEAL");
                    if (oPrice == 0) {
                        oPrice = listQDP.get(a).getOpen();
                        oPoint = a;
                    } // OPEN LONG
                    else if (listQDP.get(a).getHigh() / oPrice > 1 + (Settings.PERCENT_TAKE_PROFIT / 100)) {
                        esd = EDealStates.LONG;
                        cPrice = listQDP.get(a).getOpen();
                        cPoint = a;
                    } // OPEN SHORT
                    else if (oPrice / listQDP.get(a).getLow() > 1 + (Settings.PERCENT_TAKE_PROFIT / 100)) {
                        esd = EDealStates.SHORT;
                        cPrice = listQDP.get(a).getOpen();
                        cPoint = a;
                    }
                }; break;
                case LONG: {
                    // Поиск максимальной точки для закрытия сделки
                    if (listQDP.get(a).getHigh() > cPrice) {
                        cPrice = listQDP.get(a).getHigh();
                        cPoint = a;
                    } // Если цена опустилась на уровень предела и ниже то закрываем сделку
                    else if (cPrice / listQDP.get(a).getLow() > 1 + (Settings.PERCENT_STOP_LOSS / 100)) {
                        // Перебираем историю в поисках точки минимума
                        int op = 0;
                        for (int c = cPoint; c > oPoint; c--) {
                            if (listQDP.get(c).getOpen()< oPrice) {
                                oPrice = listQDP.get(c).getOpen();
                                op = c +1;
                            }
                        }
                        if (op != 0) oPoint = op;

                        // Собираем Dataset
                        qds = new QDataset(esd);
                        for (int l = oPoint; l <= cPoint; l++) {
                            QState qs = new QState();
                            for (int i = Settings.HISTORY_LENGHT; i >=0; i--) {
                                qs.addSeries(listQDP.get(l -i));
                            }
                            
                            if (qs.getSeries().size() == Settings.HISTORY_LENGHT +1) {
                                qds.addQState(qs);
                            }
                            else System.out.println("QState size is not equal:" + qs.getSeries().size() + " != " + (Settings.HISTORY_LENGHT +1) + ". Step: " + l);
                        }

                        // Оценка и добавление дата сета
                        qds.addReward((cPrice/oPrice -1) * 100);
                        datasets.add(qds);

                        // Обнуляем данные
                        oPrice = 0;
                        cPrice = 0;
                        oPoint = 0;
                        cPoint = 0;

                        esd = EDealStates.NODEAL;
                    }
                }; break;
                case SHORT: {
                    // Поиск минимальной точки для закрытия сделки
                    if (listQDP.get(a).getLow() < cPrice) {
                        cPrice = listQDP.get(a).getLow();
                        cPoint = a;
                    } // Если цена поднялась на уровень предела и выше то закрываем сделку
                    else if (listQDP.get(a).getHigh() / cPrice > 1 + (Settings.PERCENT_STOP_LOSS / 100)) {
                        // Перебираем историю в поисках точки максимума
                        int op = 0;
                        for (int c = cPoint; c > oPoint; c--) {
                            if (listQDP.get(c).getOpen() > oPrice) {
                                oPrice = listQDP.get(c).getOpen();
                                op = c +1;
                            }
                        }
                        if (op != 0) oPoint = op;

                        // Собираем Dataset
                        qds = new QDataset(esd);
                        for (int l = oPoint; l <= cPoint; l++) {
                            QState qs = new QState();
                            for (int i = Settings.HISTORY_LENGHT; i >=0; i--) {
                                qs.addSeries(listQDP.get(l -i));
                            }
                            
                            if (qs.getSeries().size() == Settings.HISTORY_LENGHT +1) {
                                qds.addQState(qs);
                            }
                            else System.out.println("QState size is not equal:" + qs.getSeries().size() + " != " + (Settings.HISTORY_LENGHT +1) + ". Step: " + l);
                        }

                        // Оценка и добавление дата сета
                        qds.addReward((cPrice/oPrice -1) * 100);
                        datasets.add(qds);
                        
                        // Обнуляем данные
                        oPrice = 0;
                        cPrice = 0;
                        oPoint = 0;
                        cPoint = 0;

                        esd = EDealStates.NODEAL;
                    }
                }; break;
            }
        }
    }
    
    //////////////////////////////////////////////////////////////////////////// collect_test_ds
    // Создаем тестовые принятия решений для проверки датасетов на графике торгового терминала
    public void collect_test_ds() {
        for (QDataset qds: datasets) {
            qds.calculatEval();
            /*System.out.printf("%.4f", qds.getReward());
            if (qds.getReward() != 0) {
                for (QState qs: qds.getQStates()) {
                    System.out.print(qs.getSeries().get(0).getTime() + " ... " + qs.getSeries().get(qs.getSeries().size() -1).getTime());
                    System.out.printf("buy: %.4f \t", qs.getRewBuy());
                    System.out.printf("sell: %.4f \t", qs.getRewSell());
                    System.out.printf("wait: %.4f \t", qs.getRewWait());
                    System.out.printf("close: %.4f \t", qs.getRewClose());

                    System.out.println();
                }
            }*/
        }
        
        
        for (QDataset qds: datasets) {
            if (qds.getState().equals(EDealStates.LONG)) {
                buy_test.put(
                    qds.getQStates().get(0).getSeries().get(Settings.HISTORY_LENGHT).getTime(), 
                    qds.getQStates().get(qds.getQStates().size() -1).getSeries().get(Settings.HISTORY_LENGHT).getTime()
                );
            }
        }
        for (QDataset qds: datasets) {
            if (qds.getState().equals(EDealStates.SHORT)) {
                sell_test.put(
                    qds.getQStates().get(0).getSeries().get(Settings.HISTORY_LENGHT).getTime(), 
                    qds.getQStates().get(qds.getQStates().size() -1).getSeries().get(Settings.HISTORY_LENGHT).getTime()
                );
            }
        }
    }
    
    // Остановить агента
    public void stop () {
        isStoped = true;
    }
    
}
