/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package eva.trading;

import java.io.File;

/**
 *
 * @author EVA
 */
public class Settings {
    public final static String DATA_BASE = "mt5_sm";
    public final static String DATA_BASE_DRIVER = "com.mariadb.jdbc.Driver";
    public final static String DB_SERVER = "jdbc:mariadb://localhost:3306";
    public final static String DB_USER = "root";
    public final static String DB_PASS = "[htydfv1303";
    
    // Периодичность проверок Агентом в цикле работы
    public final static long AGENT_CYCLE_PERIOD_SEC = 1;
    
    // Тестовый запуск (Проверка датасетов)
    public final static boolean IS_TEST_DS_LAUNCH = false;
    
    // *** НАСТРОЙКИ ОБУЧЕНИЯ *** //
    // Коефициент скорости обучения сети
    public final static double LEARNING_RATE = 0.00025;
    // Коефициент регуляризации l1, l2
    public final static double REGULARIZATION = 0.0;
    // Размер BATCH подаваемый при обучении сети
    public final static int BATCH_TRAINING = 256;
    // Кол-во попыток обучения нейронной сети (в случае не обучаемости)
    public final static int LIMIT_LEARNING_TRY = 3;
    // Стандартное обучение по размеченным данным
    public final static boolean FIRST_LEARNING_LAUNCH = true;
    // Минимальное движение для открытия сделки
    public final static double PERCENT_TAKE_PROFIT = 5.0;
    // Минимальное движение для закрытия сделки
    public final static double PERCENT_STOP_LOSS = 1.0;
    // Процент взятый за еденицу для вывода сети
    public final static double MAX_TAKE_PERCENT = 10;  // => +1 output = +10%
    
    // Обучение по алгоритму QLEARNING
    public final static boolean QLEAGNING_LAUNCH = false;
    // Глубина исторических данных для ввода в нейронную сеть
    public final static int HISTORY_LENGHT = 5 * 12;           // 1 торговый день = 105 свечей
    // Максимальная кол-во последних сделок для переобучения сети
    public final static int HISTORY_DEEP_QLEARNING = 100;
    // Частота включения процесса переобучения (измеряется к кол-ве завершенных сделок или кол-во свечей без сделок)
    public final static int FREQUENCY_QLEARNING = 50;
    // Штраф за простой
    public final static double DOWNTIME_PENALTY = -0.096;
    
    // *** РАНДОМНОСТЬ *** //
    public final static boolean RANDOM_LAUNCH = false;
    // Процент решений принятых рандомно (Полезно для поиска новых стратегий при обучении)
    public final static int RANDOM_PERCENT = 1;
    // Диапазон шагов рандомной сделки (Кол-во обпределяется рандомно и сделка не закрывается пока кол-во шаго не будет пройдено)
    public final static int RANDOM_STEPS_MIN = 6;
    public final static int RANDOM_STEPS_MAX = 105;
    
    
    // *** НАСТРОЙКИ НЕЙРОННОЙ СЕТИ *** //
    // Кол-во циклов обучения нейронной сети
    public final static int CYCLE_FIT_NEURON_NET = 10; 
    // Процент тестовых данных
    public final static int TEST_DATA_PERCENT = 20;
    // Кол-во DATASET'ов собранных для обучения нейронной сети
    public final static int MAX_QUANTITY_DATASETS = 250;
    // Минимальная длина сделки в свечах
    public final static int MIN_DATASET_SIZE = 3;
    // Максимальная длина сделки в свечах
    public final static int MAX_DATASET_SIZE = 210;
    
    // Путь хранениния нейронных сетей
    public final static String PATH_NEURON_NET = System.getProperty("user.dir") + File.separator +  "multi_neunet" + File.separator;
    
}
