/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package eva.trading.sql;

import eva.trading.Settings;
import eva.trading.data.EDealStates;
import eva.trading.data.EQueryStatuses;
import eva.trading.data.QDataPiece;
import eva.trading.data.QDataset;
import eva.trading.data.QState;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;


/**
 *
 * @author username
 */
public class SQLQuery {
    
    private Statement stmt;
    
    public SQLQuery() {
        
        try {
            // MariaDB
            Class.forName(Settings.DATA_BASE_DRIVER);
        } catch (ClassNotFoundException ex) {
            System.out.println(SQLQuery.class.getName() + " -> " + ex);
        }
        
        // Создаем подключение к базе
        try {
            stmt = DriverManager.getConnection(Settings.DB_SERVER + "/" + Settings.DATA_BASE + "?useSSL=false", Settings.DB_USER, Settings.DB_PASS).createStatement();
        }
        catch (Exception e) {
            stmt = null;
        }
    }
    
    // Состояние сделки (LONG, SHORT, NODEAL)
    public synchronized EDealStates getStateByTime(String tbl, long timestamp) {
        try {
            stmt = getConnection();
            
            ResultSet rs = stmt.executeQuery("SELECT * FROM `" + tbl + "_exchange` WHERE `timestamp`=" + timestamp + ";");
            if(rs.next()) {
                return EDealStates.valueOf(rs.getString("state"));
            }
        } catch (SQLException ex) {
            System.out.println(SQLQuery.class.getName() + " -> " + ex);
        }
        return null;
    }
    
    // Отметки времени по статусу запроса
    public synchronized List<Long> getTimestampsByState(String tbl, EQueryStatuses eqs) {
        List<Long> ts = new ArrayList<>();
        
        try {
            stmt = getConnection();
            
            ResultSet rs = stmt.executeQuery("SELECT `timestamp` FROM `" + tbl + "_exchange` WHERE `status`='" + eqs.toString() + "' ORDER BY `timestamp`;");
            
            while(rs.next()) {
                ts.add(new Long(rs.getLong("timestamp")));
            }
        } catch (SQLException ex) {
            System.out.println(SQLQuery.class.getName() + " -> " + ex);
        }
        
        return ts;
    }
    
    
    // Сбор исторических данных до отметки времени (порядок хронологический)
    public synchronized QState getQState(String tbl, long timestamp, int deepHistory) {
        
        QState qst = new QState();
        
        try {
            stmt = getConnection();
            
            ResultSet rs = stmt.executeQuery("SELECT * FROM `" + tbl + "` WHERE `timestamp` < " + timestamp + " ORDER BY `timestamp` DESC LIMIT " + deepHistory + ";");
            
            while (rs.next()) {
                QDataPiece qdp = new QDataPiece(
                    rs.getLong("timestamp"), 
                    rs.getDouble("open"), 
                    rs.getDouble("close"), 
                    rs.getDouble("high"), 
                    rs.getDouble("low"), 
                    rs.getDouble("volume"),
                        
                    rs.getDouble("rsi_3"), 
                    rs.getDouble("rsi_7"), 
                    rs.getDouble("rsi_13"), 
                    rs.getDouble("rsi_21"), 
                    rs.getDouble("rsi_31"), 
                        
                    rs.getDouble("stoh_m_3"),
                    rs.getDouble("stoh_m_7"),
                    rs.getDouble("stoh_m_13"),
                    rs.getDouble("stoh_m_21"),
                    rs.getDouble("stoh_m_31"),
                        
                    rs.getDouble("stoh_s_3"),
                    rs.getDouble("stoh_s_7"),
                    rs.getDouble("stoh_s_13"),
                    rs.getDouble("stoh_s_21"),
                    rs.getDouble("stoh_s_31"),
                        
                    rs.getDouble("adx_3"),
                    rs.getDouble("adx_7"),
                    rs.getDouble("adx_13"),
                    rs.getDouble("adx_21"),
                    rs.getDouble("adx_31"),
                    
                    rs.getDouble("dip_3"),
                    rs.getDouble("dip_7"),
                    rs.getDouble("dip_13"),
                    rs.getDouble("dip_21"),
                    rs.getDouble("dip_31"),
                        
                    rs.getDouble("dim_3"),
                    rs.getDouble("dim_7"),
                    rs.getDouble("dim_13"),
                    rs.getDouble("dim_21"),
                    rs.getDouble("dim_31"),
                        
                    rs.getDouble("ema_3"),
                    rs.getDouble("ema_7"),
                    rs.getDouble("ema_13"),
                    rs.getDouble("ema_21"),
                    rs.getDouble("ema_31")
                );
                qst.addSeries(qdp);
            }
            
        } catch (SQLException ex) {
            System.out.println(SQLQuery.class.getName() + " -> " + ex);
            return null;
        } 
        
        // Пересобираем из обратного порядка в хронологический
        Collections.reverse(qst.getSeries());
        
        return qst;
    }
    
    // Сбор исторических данных до отметки времени (порядок хронологический)
    public synchronized List <QDataPiece> getAllHistory(String tbl) {
        List <QDataPiece> listQDP = new ArrayList<>();
        
        try {
            stmt = getConnection();
            
            ResultSet rs = stmt.executeQuery("SELECT * FROM `" + tbl + "` ORDER BY `timestamp` ASC;");
            
            while (rs.next()) {
                QDataPiece qdp = new QDataPiece(
                    rs.getLong("timestamp"), 
                    rs.getDouble("open"), 
                    rs.getDouble("close"), 
                    rs.getDouble("high"), 
                    rs.getDouble("low"), 
                    rs.getDouble("volume"),
                        
                    rs.getDouble("rsi_3"), 
                    rs.getDouble("rsi_7"), 
                    rs.getDouble("rsi_13"), 
                    rs.getDouble("rsi_21"), 
                    rs.getDouble("rsi_31"), 
                        
                    rs.getDouble("stoh_m_3"),
                    rs.getDouble("stoh_m_7"),
                    rs.getDouble("stoh_m_13"),
                    rs.getDouble("stoh_m_21"),
                    rs.getDouble("stoh_m_31"),
                        
                    rs.getDouble("stoh_s_3"),
                    rs.getDouble("stoh_s_7"),
                    rs.getDouble("stoh_s_13"),
                    rs.getDouble("stoh_s_21"),
                    rs.getDouble("stoh_s_31"),
                        
                    rs.getDouble("adx_3"),
                    rs.getDouble("adx_7"),
                    rs.getDouble("adx_13"),
                    rs.getDouble("adx_21"),
                    rs.getDouble("adx_31"),
                    
                    rs.getDouble("dip_3"),
                    rs.getDouble("dip_7"),
                    rs.getDouble("dip_13"),
                    rs.getDouble("dip_21"),
                    rs.getDouble("dip_31"),
                        
                    rs.getDouble("dim_3"),
                    rs.getDouble("dim_7"),
                    rs.getDouble("dim_13"),
                    rs.getDouble("dim_21"),
                    rs.getDouble("dim_31"),
                        
                    rs.getDouble("ema_3"),
                    rs.getDouble("ema_7"),
                    rs.getDouble("ema_13"),
                    rs.getDouble("ema_21"),
                    rs.getDouble("ema_31")
                );
                listQDP.add(qdp);
            }
            
        } catch (SQLException ex) {
            System.out.println(SQLQuery.class.getName() + " -> " + ex);
            return null;
        } 
        
        return listQDP;
    }
    
    // Получить список временых меток
    public synchronized List <Long> getTimestamps(String tbl, long since, long till) {
        List<Long> ts = new ArrayList<>();
        
        try {
            stmt = getConnection();
            
            ResultSet rs = stmt.executeQuery("SELECT `timestamp` FROM `" + tbl + "` WHERE `timestamp`>=" + since + " AND `timestamp`<=" + till + " ORDER BY `timestamp` DESC;");
            
            while (rs.next()) {
                ts.add(new Long(rs.getLong("timestamp")));
            }
        } catch (SQLException ex) {
            System.out.println(SQLQuery.class.getName() + " -> " + ex);
        }
        
        // Пересобираем из обратного порядка в хронологический
        Collections.reverse(ts);
        
        return ts;
    }
    
    // Получить список временых меток о закрытии позиции (порядок хронологический)
    public synchronized List <Long> getTimestampsClosingDeal(String tbl, int deepHistory) {
        List<Long> ts = new ArrayList<>();
        
        try {
            stmt = getConnection();
            
            ResultSet rs = stmt.executeQuery("SELECT `timestamp_close` FROM `" + tbl + "_exchange` WHERE `timestamp_close`>0 ORDER BY `timestamp_close` DESC LIMIT " + deepHistory + ";");
            
            while (rs.next()) {
                ts.add(new Long(rs.getLong("timestamp_close")));
            }
        } catch (SQLException ex) {
            System.out.println(SQLQuery.class.getName() + " -> " + ex);
        }
        
        // Пересобираем из обратного порядка в хронологический
        Collections.reverse(ts);
        
        return ts;
    }
    
    // Получить список временых меток о закрытии позиции по типу сделки(порядок хронологический)
    public synchronized List <Long> getTimestampsClosingDealByDealType(String tbl, EDealStates eds, int deepHistory) {
        List<Long> ts = new ArrayList<>();
        
        try {
            stmt = getConnection();
            
            ResultSet rs = stmt.executeQuery("SELECT `timestamp_close` FROM `" + tbl + "_exchange` WHERE `timestamp_close`>0 AND `state`='" + eds + "' ORDER BY `timestamp_close` DESC LIMIT " + deepHistory + ";");
            
            while (rs.next()) {
                ts.add(new Long(rs.getLong("timestamp_close")));
            }
        } catch (SQLException ex) {
            System.out.println(SQLQuery.class.getName() + " -> " + ex);
        }
        
        // Пересобираем из обратного порядка в хронологический
        Collections.reverse(ts);
        
        return ts;
    }
    
    // Получить датасет по времени закрытия сделки
    public synchronized QDataset getQDatasetByCloseTime(String tbl, long closeTime) {
        QDataset qds = null;
        List <Long> tsList;
        
        try {
            stmt = getConnection();
            
            // Запрос типа сделки
            ResultSet rs = stmt.executeQuery("SELECT * FROM `" + tbl + "_exchange` WHERE `timestamp_close`=" + closeTime + ";");
            if(rs.next()) {
                qds = new QDataset(EDealStates.valueOf(rs.getString("state")));
                double eval = rs.getDouble("evalute");
                if (eval < 0) {
                    qds.addReward(rs.getDouble("evalute") / Settings.PERCENT_STOP_LOSS);
                }
                else {
                    qds.addReward(rs.getDouble("evalute") / Settings.MAX_TAKE_PERCENT);
                }
                
                tsList = getTimestamps(tbl, rs.getLong("timestamp"), rs.getLong("timestamp_close")); 
                for (int t = 0;  t < tsList.size(); t++) {
                    qds.addQState(getQState(tbl, tsList.get(t), Settings.HISTORY_LENGHT + 1));
                }
            }
        } catch (SQLException ex) {
            System.out.println(SQLQuery.class.getName() + " -> " + ex);
        }
        
        return qds;
    }
    
    // Получить датасет по времени закрытия сделки
    public synchronized List <QDataset> getQDatasetForWait(String tbl, int deepHistory) {
        List <QDataset> qdsList = new ArrayList<>();
        
        try {
            stmt = getConnection();
            
            //Собирае
            ResultSet rs = stmt.executeQuery("SELECT * FROM `" + tbl + "_exchange` WHERE `status`=\"" + EQueryStatuses.WAIT + "\" AND `state`=\"" + EDealStates.NODEAL + "\" ORDER BY `timestamp` DESC LIMIT " + deepHistory + ";");
            while(rs.next()) {
                QDataset qds = new QDataset(EDealStates.NODEAL);
                qds.addQState(getQState(tbl, rs.getLong("timestamp"), Settings.HISTORY_LENGHT + 1));
                qdsList.add(qds);
            }
        } catch (SQLException ex) {
            System.out.println(SQLQuery.class.getName() + " -> " + ex);
        }
        
        return qdsList;
    }
    
    
    // Запись ответа в таблицу обмена
    public void setStatus(String tbl, EQueryStatuses eqs, long timestamp)  {
        try{
            stmt = getConnection();
            
            stmt.execute("UPDATE `" + tbl + "_exchange` SET `status`='" + eqs.toString() + "' WHERE `timestamp`=" + timestamp + ";");
        } catch (SQLException ex) {
            System.out.println(SQLQuery.class.getName() + " -> " + ex);
        } 
    }
    
    // Запись прогноза в таблицу обмена
    public void setForecast(String tbl, double forecast, long timestamp)  {
        try{
            stmt = getConnection();
            
            stmt.execute("UPDATE `" + tbl + "_exchange` SET `forecast`=" + forecast + " WHERE `timestamp`=" + timestamp + ";");
        } catch (SQLException ex) {
            System.out.println(SQLQuery.class.getName() + " -> " + ex);
        } 
    }
    
    
    // Проверяем живое ли подключение к БД, если нет, то переподключаемся
    private Statement getConnection () {
        
        for (int i = 0; i < 5; i++) {
            
            try {
                if (stmt == null) {
                    try {
                        stmt = DriverManager.getConnection(Settings.DB_SERVER + "/" + Settings.DATA_BASE + "?useSSL=false", Settings.DB_USER, Settings.DB_PASS).createStatement();
                        return stmt;
                    }
                    catch (Exception e) {
                        stmt = null;
                    }
                }
                else if (stmt.isClosed()) {
                    try {
                        stmt = DriverManager.getConnection(Settings.DB_SERVER + "/" + Settings.DATA_BASE + "?useSSL=false", Settings.DB_USER, Settings.DB_PASS).createStatement();
                        return stmt;
                    }
                    catch (Exception e) {
                        stmt = null;
                    }
                } 
                else return stmt;
                
            } catch (SQLException ex) {
                System.out.println(ex);
                continue;
            }
            
            try {
                Thread.sleep(1000);
            } catch (InterruptedException ex) {
                Logger.getLogger(SQLQuery.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        
        return null;
    }
    
}
