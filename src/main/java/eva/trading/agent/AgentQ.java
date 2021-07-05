/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package eva.trading.agent;

import eva.trading.data.QState;
import eva.trading.data.QDataset;
import eva.trading.data.EDealStates;
import eva.trading.data.EQueryStatuses;
import eva.trading.Settings;
import eva.trading.nn.MultiDeepNeuralNetwork_4D;
import eva.trading.sql.SQLQuery;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.GregorianCalendar;
import java.util.Date;
import java.util.List;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
//import org.sikuli.script.FindFailed;

//import org.sikuli.script.Screen;

/**
 *
 * @author EVA
 * Агент имеет 3 нейроных сети,
 * каждая сеть предназначена для 1го из 3х действий
 * BUY, SELL, WAIT
 * Каждая сетьб переобучается отдельно от остальных после каждой сделки
 * 
 * Функция активации ТАНГЕНС => вывож сети в диапазоне от -1 до 1
 * +1 Эквивалентен предполагаемому изменению цены в нашу пользу на 10%
 * -1 Эквивалентен предполагаемому изменению цены не в нашу пользу на 1%
 * Убыточная сделка более критична, чем выгодная => соотношение 1 к 10
 */
public class AgentQ implements Runnable {
    
    private List <QDataset> datasets_buy    = null;          // Набор сделок
    private List <QDataset> datasets_sell   = null;          // Набор сделок
    private List <QDataset> datasets_wait   = null;          // Набор сделок
    private List <QDataset> datasets_close  = null;          // Набор сделок
    
    private MultiDeepNeuralNetwork_4D nn_buy   = null;              // Нейронная сеть агента
    private MultiDeepNeuralNetwork_4D nn_sell  = null;              // Нейронная сеть агента
    private MultiDeepNeuralNetwork_4D nn_wait  = null;              // Нейронная сеть агента
    private MultiDeepNeuralNetwork_4D nn_close = null;              // Нейронная сеть агента
    
    private SQLQuery sql;
    
    private boolean isStoped = false;
    
    private String symbol;
    // Является ли текущая сделка рандомной
    private boolean isRandomDial = false;
    // Счетчик сделок
    private int counterDeals = 0;
    // Счетчик простоя
    private int downtime_counter = 0;
    // Показатели нейронной сети
    private double toBuy = 0;
    private double toSell = 0;
    private double toWait = 0;
    private double toClose = 0;
    // Захват монитора
    //private Screen s;
    // Календарь
    private Calendar gs;
    // Рандомайзер
    private Random r;
    // Счетчик пройденных сделок в рандомной сделке (пока счетчик не истечёт, сделка не будет завершена)
    private int random_waiting_counter; 
    
    public AgentQ (String symbol_tab) throws IOException {
        
        symbol = symbol_tab;
        
        datasets_buy    = new ArrayList <>();
        datasets_sell   = new ArrayList <>();
        datasets_wait   = new ArrayList <>();
        datasets_close  = new ArrayList <>();
        
        nn_buy      = loadMNN(EQueryStatuses.BUY);
        nn_sell     = loadMNN(EQueryStatuses.SELL);
        nn_wait     = loadMNN(EQueryStatuses.WAIT);
        nn_close    = loadMNN(EQueryStatuses.CLOSE);
        
        // Создаем подключение к SQL
        sql = new SQLQuery();
        // Берём изображение монитора
        //s = new Screen();
        
        gs = new GregorianCalendar();
        
        r = new Random();
        
        // Запускаем поток агента
        new Thread(this).start();
    }
    
    //////////////////////////////////////////////////////////////////////////// RUN
    @Override
    public void run() {
        System.out.println("Trading system is started.");
        
        boolean conditionDials = false;
        
        // Цикл работы агента
        long newCheck = new Date().getTime();
        while (!isStoped) {
            
            if (newCheck <= new Date().getTime()) {
                newCheck += Settings.AGENT_CYCLE_PERIOD_SEC;
                // Если время последнего запроса не совпадает с последним запросом от MT5 => Пришёл новый запрос
                List <Long> queries = sql.getTimestampsByState(symbol, EQueryStatuses.QUERY);
                // Расчитываем прогноз и пишем его в БД
                for (long timestamp: queries) {
                    System.out.println("TIMESTAMP: \t" + timestamp);
                    
                    QState qs = sql.getQState(symbol, timestamp, Settings.HISTORY_LENGHT + 1);

                    EDealStates eds = sql.getStateByTime(symbol, timestamp);
                    EQueryStatuses eqs = null;
                    
                    toBuy  = nn_buy.prediction(qs, eds);
                    System.out.printf("BUY:  %.4f%n",toBuy);
                    toSell = nn_sell.prediction(qs, eds);
                    System.out.printf("SELL: %.4f%n",toSell);
                    toWait = nn_wait.prediction(qs, eds);
                    System.out.printf("WAIT: %.4f%n",toWait);
                    toClose = nn_close.prediction(qs, eds);
                    System.out.printf("CLOSE: %.4f%n",toClose);
                    
                    switch (eds) {
                        // Если мы находимся в сделке ЛОНГ
                        case LONG: {
                            if (isRandomDial) {
                                random_waiting_counter--;
                                if (random_waiting_counter == 0) {
                                    eqs = EQueryStatuses.CLOSE;
                                    isRandomDial = false;
                                }
                                else eqs = EQueryStatuses.WAIT;
                            }
                            else if (toClose > toWait) {
                                eqs = EQueryStatuses.CLOSE;
                            }
                            else {
                                eqs = EQueryStatuses.WAIT;
                            }
                        } break;
                        // Если мы находимся в сделке ЛОНГ
                        case SHORT: {
                            if (isRandomDial) {
                                random_waiting_counter--;
                                if (random_waiting_counter == 0) {
                                    eqs = EQueryStatuses.CLOSE;
                                    isRandomDial = false;
                                }
                                else eqs = EQueryStatuses.WAIT;
                            }
                            else if (toClose > toWait) {
                                eqs = EQueryStatuses.CLOSE;
                            }
                            else {
                                eqs = EQueryStatuses.WAIT;
                            }
                        } break;
                        // ЕСЛИ У НАС НЕТ СДЕЛКИ
                        default: {
                            // Рандомное решение
                            if (Settings.RANDOM_LAUNCH) {
                                random_waiting_counter = 0;
                                isRandomDial = false;
                                
                                if (r.nextInt(100 / (Settings.RANDOM_PERCENT /* random_dec*/)) == 1) {
                                    random_waiting_counter = Settings.RANDOM_STEPS_MIN + r.nextInt(Settings.RANDOM_STEPS_MAX - Settings.RANDOM_STEPS_MIN);
                                    if(r.nextInt(2) == 0) { 
                                        eqs = EQueryStatuses.SELL;
                                    }
                                    else {
                                        eqs = EQueryStatuses.BUY;
                                    }
                                    isRandomDial = true;
                                }
                            }
                            // Решение нейронной сети
                            else if (toBuy > toSell && toBuy > toWait)  eqs = EQueryStatuses.BUY;
                            else if (toSell > toBuy && toSell > toWait) eqs = EQueryStatuses.SELL;
                            else eqs = EQueryStatuses.WAIT;
                        }
                        
                        // Счетчик простоя, при переполнении, переобучение
                        if (eds.equals(EDealStates.NODEAL) && eqs.equals(EQueryStatuses.WAIT)) {
                            downtime_counter++;
                        }
                        
                    }
                    sql.setStatus(symbol, eqs, timestamp);
                    
                    if (eqs.equals(EQueryStatuses.BUY) || eqs.equals(EQueryStatuses.SELL)) {
                        conditionDials = true;
                    }
                }
            }
            
            if (conditionDials)  {
                counterDeals++;
                conditionDials = false;
            }
            
            // Переобучение в Начале СУТОК (Для РАБОТЫ)
            // gs.setTime(new Date());
            //if (gs.get(Calendar.HOUR) == 00 && gs.get(Calendar.MINUTE) == 5) {
            // Переобучение по счётчику сделок или по счетчику переполнения (Для теста)
            if (Settings.QLEAGNING_LAUNCH && (counterDeals > Settings.FREQUENCY_QLEARNING || downtime_counter > Settings.FREQUENCY_QLEARNING)) {
            // Переобучение нейронной сети
                /*try {
                    counterDeals = 0;
                    // Приостанавливаем MT5
                    s.click(Settings.PATH_NEURON_NET + "/pause.png");
                    s.wait(2.0);
                    s.click(Settings.PATH_NEURON_NET + "/nothing.png");
                    
                    toCollectDatasets();
                    toLearn();
                    downtime_counter = 0;
                    
                    // Запускаем MT5
                    s.click(Settings.PATH_NEURON_NET + "/play.png");
                    s.wait(2.0);
                    s.click(Settings.PATH_NEURON_NET + "/nothing.png");
                } catch (FindFailed ex) {
                    System.out.println(ex);
                    //Logger.getLogger(AgentQ.class.getName()).log(Level.SEVERE, null, ex);
                }*/
            }
            
            try {
                Thread.sleep(100); //
            } catch (InterruptedException ex) {
                Logger.getLogger(AgentQ.class.getName()).log(Level.SEVERE, null, ex);
            }
        } 
    }
    
    //////////////////////////////////////////////////////////////////////////// LOAD MNN
    // Загрузка нейронной сети
    private MultiDeepNeuralNetwork_4D loadMNN(EQueryStatuses name) throws IOException {
        MultiDeepNeuralNetwork_4D nn;
        
        if (new File(Settings.PATH_NEURON_NET + symbol + name.toString()).exists()) {
            nn = new MultiDeepNeuralNetwork_4D(Settings.HISTORY_LENGHT, Settings.PATH_NEURON_NET + symbol + name.toString(), name);
            System.out.println("Neural Network was read from file: \"" + Settings.PATH_NEURON_NET + symbol + name.toString() + "\"");
        }
        else nn = new MultiDeepNeuralNetwork_4D (Settings.HISTORY_LENGHT, name);
        File f = new File(Settings.PATH_NEURON_NET);
        try{
            if(f.mkdir()) { 
                System.out.println("Directory \"" + Settings.PATH_NEURON_NET + symbol + name.toString() + "\" was created.");
            } 
        } catch(Exception e){
            e.printStackTrace();
        } 
        
        return nn;
    }
    
    //////////////////////////////////////////////////////////////////////////// toCollectDatasets
    // Алгорит сбора выборок из опыта для дообучения
    public void toCollectDatasets() {
        // Очищаем список датасетов
        datasets_buy.clear();
        datasets_sell.clear();
        datasets_wait.clear();
        datasets_close.clear();
        
        List <Long> tscList;
        
        // Перебираем отметки и собираем Dataset'ы (LONG, SHORT, WAIT)
        System.out.println("DATASET IS COLLECTING...");
        // Получаем список отметок времени о закрытии сделок LONG
        tscList = sql.getTimestampsClosingDealByDealType(symbol, EDealStates.LONG, Settings.HISTORY_DEEP_QLEARNING);
        for (int l = 0; l < tscList.size(); l++) {
            QDataset qds = sql.getQDatasetByCloseTime(symbol, tscList.get(l));
            if (qds != null) datasets_buy.add(qds);
        }
        System.out.println("DATASET `LONG` WAS COLLECTED: " + datasets_buy.size());
        // Получаем список отметок времени о закрытии сделок SHORT
        tscList = sql.getTimestampsClosingDealByDealType(symbol, EDealStates.SHORT, Settings.HISTORY_DEEP_QLEARNING);
        for (int l = 0; l < tscList.size(); l++) {
            QDataset qds = sql.getQDatasetByCloseTime(symbol, tscList.get(l));
            if (qds != null) datasets_sell.add(qds);
        }
        System.out.println("DATASET `SHORT` WAS COLLECTED: " + datasets_sell.size());
        // Получаем список отметок времени о простоях
        for (QDataset qds: sql.getQDatasetForWait(symbol, (datasets_buy.size() + datasets_sell.size()))) {
            if (qds != null) datasets_wait.add(qds);
        }
        datasets_wait.addAll(datasets_buy);
        datasets_wait.addAll(datasets_sell);
        System.out.println("DATASET `NODEAL` WAS COLLECTED: " + datasets_wait.size());
        // Создаем список для закрытия
        datasets_close.addAll(datasets_buy);
        datasets_close.addAll(datasets_sell);
        System.out.println("DATASET `CLOSE` WAS COLLECTED: " + datasets_close.size());
    } 
    
    //////////////////////////////////////////////////////////////////////////// toLearn
    // ОБУЧЕНИЕ
    public void toLearn() {
        
        boolean isTrainingBuy = false;
        boolean isTrainingSell = false;
        boolean isTrainingWait = false;
        
        if (datasets_buy.size() > 1) {
            boolean isTest = false;
            System.out.println("PUT DATA TO nn_buy.");
            for (int d = 0; d < datasets_buy.size(); d++) {
                datasets_buy.get(d).calculatEval();
                if (datasets_buy.get(d).getState().equals(EDealStates.LONG)) {
                    if (isTest) {
                        nn_buy.putTrainData(datasets_buy.get(d));
                        isTest = !isTest;
                    }
                    else {
                        nn_buy.putTestData(datasets_buy.get(d));
                        isTest = !isTest;
                        isTrainingBuy = true;
                    }
                }
            }
        }
        if (datasets_sell.size() > 1) {
            boolean isTest = false;
            System.out.println("PUT DATA TO nn_sell.");
            for (int d = 0; d < datasets_sell.size(); d++) {
                datasets_sell.get(d).calculatEval();
                if (datasets_sell.get(d).getState().equals(EDealStates.SHORT)) {
                    if (isTest) {
                        nn_sell.putTrainData(datasets_sell.get(d));
                        isTest = !isTest;
                    }
                    else {
                        nn_sell.putTestData(datasets_sell.get(d));
                        isTest = !isTest;
                        isTrainingSell = true;
                    }
                }
            }
        }
        if (datasets_wait.size() > 1) {
            boolean isTest = false;
            System.out.println("PUT DATA TO nn_wait.");
            for (int d = 0; d < datasets_wait.size(); d++) {
                datasets_wait.get(d).calculatEval();
                if (datasets_wait.get(d).getState().equals(EDealStates.NODEAL)) {
                    if (isTest) {
                        nn_wait.putTrainData(datasets_wait.get(d));
                        isTest = !isTest;
                    }
                    else {
                        nn_wait.putTestData(datasets_wait.get(d));
                        isTest = !isTest;
                        isTrainingWait = true;
                    }
                }
            }
        }
        if (datasets_close.size() > 1) {
            boolean isTest = false;
            System.out.println("PUT DATA TO nn_close.");
            for (int d = 0; d < datasets_close.size(); d++) {
                datasets_close.get(d).calculatEval();
                if (datasets_close.get(d).getState().equals(EDealStates.NODEAL)) {
                    if (isTest) {
                        nn_close.putTrainData(datasets_close.get(d));
                        isTest = !isTest;
                    }
                    else {
                        nn_close.putTestData(datasets_close.get(d));
                        isTest = !isTest;
                        isTrainingWait = true;
                    }
                }
            }
        }
        
        
        // Обучение сети отвечающей за BUY, если обучение требуется
        if (isTrainingBuy) {
            // ПОДГОТОВКА ДАНЫХ
            nn_buy.prepareDatasets();

            // ТЕСТ И СБОР ОЦЕНОК
            nn_buy.testNetworkRegression();
            double mqe = nn_buy.mqe();

            // ОБУЧЕНИЕ
            System.out.println("NEURAL NETWORK `BUY` IS TRAINING...");
            for (int f = 0; f < 10; f++) {
                nn_buy.trainingNetwork((int)Settings.CYCLE_FIT_NEURON_NET / 10);
                System.out.println("COMPLETED BY " + ((f+1) * 10) + "%");
            }
            System.out.println("TRAINING COMPLETED.");

            // ТЕСТ И СБОР ОЦЕНОК ПОСЛЕ ОБУЧЕНИЯ
            nn_buy.testNetworkRegression();
            if (mqe > nn_buy.mqe()) {
                try {
                    nn_buy.saveNeuralNetwork(Settings.PATH_NEURON_NET + symbol + "_buy");
                    System.out.println("NEURAL NETWORK SAVED TO: " + Settings.PATH_NEURON_NET + symbol + "_buy");
                } catch (IOException ex) {
                    Logger.getLogger(AgentQ.class.getName()).log(Level.SEVERE, null, ex);
                }
            }
            else {
                System.out.println("NEURAL NETWORK `BUY` DIDN'T SAVED 'CAUSE NEW RMSE BIGGER THAN OLD.");
            }
        }
        
        // Обучение сети отвечающей за SELL, если обучение требуется
        if (isTrainingSell) {
            // ПОДГОТОВКА ДАНЫХ
            nn_sell.prepareDatasets ();

            // ТЕСТ И СБОР ОЦЕНОК
            nn_sell.testNetworkRegression();
            double mqe = nn_sell.mqe();

            // ОБУЧЕНИЕ
            System.out.println("NEURAL NETWORK `SELL` IS TRAINING...");
            for (int f = 0; f < 10; f++) {
                nn_sell.trainingNetwork((int)Settings.CYCLE_FIT_NEURON_NET / 10);
                System.out.println("COMPLETED BY " + ((f+1) * 10) + "%");
            }
            System.out.println("TRAINING COMPLETED.");

            // ТЕСТ И СБОР ОЦЕНОК ПОСЛЕ ОБУЧЕНИЯ
            nn_sell.testNetworkRegression();
            if (mqe > nn_sell.mqe()) {
                try {
                    nn_sell.saveNeuralNetwork(Settings.PATH_NEURON_NET + symbol + "_sell");
                    System.out.println("NEURAL NETWORK SAVED TO: " + Settings.PATH_NEURON_NET + symbol + "_sell");
                } catch (IOException ex) {
                    Logger.getLogger(AgentQ.class.getName()).log(Level.SEVERE, null, ex);
                }
            }
            else {
                System.out.println("NEURAL NETWORK `SELL` DIDN'T SAVED 'CAUSE NEW RMSE BIGGER THAN OLD.");
            }
        }
        
        // Обучение сети отвечающей за WAIT, если обучение требуется
        if (isTrainingWait) {
            // ПОДГОТОВКА ДАНЫХ
            nn_wait.prepareDatasets ();

            // ТЕСТ И СБОР ОЦЕНОК
            nn_wait.testNetworkRegression();
            double mqe = nn_wait.mqe();

            // ОБУЧЕНИЕ
            System.out.println("NEURAL NETWORK `WAIT` IS TRAINING...");
            for (int f = 0; f < 10; f++) {
                nn_wait.trainingNetwork((int)Settings.CYCLE_FIT_NEURON_NET / 10);
                System.out.println("COMPLETED BY " + ((f+1) * 10) + "%");
            }
            System.out.println("TRAINING COMPLETED.");

            // ТЕСТ И СБОР ОЦЕНОК ПОСЛЕ ОБУЧЕНИЯ
            nn_wait.testNetworkRegression();
            if (mqe > nn_wait.mqe()) {
                try {
                    nn_wait.saveNeuralNetwork(Settings.PATH_NEURON_NET + symbol + "_wait");
                    System.out.println("NEURAL NETWORK SAVED TO: " + Settings.PATH_NEURON_NET + symbol + "_wait");
                } catch (IOException ex) {
                    Logger.getLogger(AgentQ.class.getName()).log(Level.SEVERE, null, ex);
                }
            }
            else {
                System.out.println("NEURAL NETWORK `WAIT` DIDN'T SAVED 'CAUSE NEW RMSE BIGGER THAN OLD.");
            }
        }
        
    }
    
    // Остановить агента
    public void stop () {
        isStoped = true;
    }
    
}
